import pandas as pd
from typing import Dict, List


# ============================================================
# BACKTEST / RISK PARAMETERS
# ============================================================

INITIAL_BALANCE = 1_000_000.0
BROKERAGE_RATE = 0.0005
MAX_PORTFOLIO_ALLOCATION = 0.25
MAX_VOLUME_PERCENT = 0.10
CASH_BUFFER_PERCENT = 0.02


# ============================================================
# TRADE RECORD
# ============================================================

def _trade_record(
    trade_id,
    entry_time,
    exit_time,
    ticker,
    direction,
    pnl,
    running_balance,
    pair_id=''
):
    return {
        'Trade_ID': trade_id,
        'Entry_Time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Exit_Time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Ticker': ticker,
        'Direction': direction,
        'Pair_ID': pair_id,
        'P&L': round(float(pnl), 2),
        'Running_Balance': round(float(running_balance), 2)
    }


# ============================================================
# RISK AND POSITION SIZING
# ============================================================

class RiskAndSizingEngine:
    """
    Apply capital, liquidity, cash-buffer and brokerage constraints.

    The volume constraint deliberately uses the signal day's volume,
    because the trade executes at the following day's Open and the
    following day's full volume is not yet known.
    """

    def calculate_quantity(
        self,
        portfolio_value: float,
        available_cash: float,
        entry_price: float,
        prior_day_volume: float,
    ) -> int:

        if entry_price <= 0 or prior_day_volume <= 0:
            return 0

        safe_cash = (
            available_cash *
            (1.0 - CASH_BUFFER_PERCENT)
        )

        capital_limit = (
            portfolio_value *
            MAX_PORTFOLIO_ALLOCATION
        )

        usable_capital = min(
            capital_limit,
            safe_cash
        )

        per_share_cost = (
            entry_price *
            (1.0 + BROKERAGE_RATE)
        )

        qty_by_capital = int(
            usable_capital // per_share_cost
        )

        qty_by_volume = int(
            prior_day_volume *
            MAX_VOLUME_PERCENT
        )

        return max(
            0,
            min(qty_by_capital, qty_by_volume)
        )


# ============================================================
# P&L HELPERS
# ============================================================

def _long_pnl(entry_price, exit_price, qty):
    entry_fee = entry_price * qty * BROKERAGE_RATE
    exit_fee = exit_price * qty * BROKERAGE_RATE

    return (
        (exit_price - entry_price) * qty
        - entry_fee
        - exit_fee
    )


def _short_pnl(entry_price, exit_price, qty):
    entry_fee = entry_price * qty * BROKERAGE_RATE
    exit_fee = exit_price * qty * BROKERAGE_RATE

    return (
        (entry_price - exit_price) * qty
        - entry_fee
        - exit_fee
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def _prepare_strategy_data(df_raw, strategy_func):
    """
    Normal strategies receive one ticker at a time.

    Pairs trading needs the entire cross-section to select a pair.
    """

    if getattr(strategy_func, 'requires_full_data', False):
        return strategy_func(
            df_raw.sort_values(
                ['dt', 'tkr']
            ).copy()
        )

    processed_dfs = []

    for _, group_df in df_raw.groupby('tkr'):
        processed_dfs.append(
            strategy_func(group_df.copy())
        )

    return pd.concat(
        processed_dfs,
        ignore_index=True
    )


# ============================================================
# PORTFOLIO VALUE
# ============================================================

def _portfolio_value(
    today_data: pd.DataFrame,
    active_positions: Dict[str, dict],
    current_cash: float
) -> float:
    """
    Mark carried long positions at today's Open.

    The Open is the actual execution price for any new trade on
    this date, so this does not use the day's Close or Volume.
    """

    value = current_cash

    for ticker, position in active_positions.items():
        row = today_data[
            today_data['tkr'] == ticker
        ]

        if not row.empty:
            value += (
                position['qty'] *
                float(row.iloc[0]['aO'])
            )

    return value


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    df_raw: pd.DataFrame,
    strategy_func
) -> pd.DataFrame:
    """
    Causal EOD backtester following the repository's execution model.

    Long:
        Signal on day i -> enter at day i+1 Open.
        Exit signal on day i -> exit at day i+1 Open.

    Short:
        Signal on day i -> enter at day i+1 Open.
        Cover at day i+1 Close.

    Pairs:
        Pair signal on day i -> both legs enter at day i+1 Open.
        Both legs close at day i+1 Close.
    """

    df = _prepare_strategy_data(
        df_raw,
        strategy_func
    )

    df = df.sort_values(
        ['dt', 'tkr']
    ).reset_index(drop=True)

    balance = INITIAL_BALANCE
    cumulative_pnl = 0.0

    active_positions = {}
    trade_history: List[dict] = []
    trade_id_counter = 1

    unique_dates = (
        df['dt']
        .sort_values()
        .unique()
    )

    # ------------------------------------------------------------
    # T+1 EXECUTION
    # ------------------------------------------------------------

    for i in range(len(unique_dates) - 1):

        today_date = unique_dates[i]
        tomorrow_date = unique_dates[i + 1]

        today_data = df[
            df['dt'] == today_date
        ].copy()

        tomorrow_data = df[
            df['dt'] == tomorrow_date
        ].copy()

        tomorrow_lookup = {
            row['tkr']: row
            for _, row in tomorrow_data.iterrows()
        }

        # ========================================================
        # A. EXIT CARRIED LONG POSITIONS
        # ========================================================

        tickers_to_remove = []

        for ticker, position in active_positions.items():

            today_row = today_data[
                today_data['tkr'] == ticker
            ]

            if today_row.empty:
                continue

            row = today_row.iloc[0]
            exit_signal = row.get(
                'Exit_Signal',
                0
            )

            if exit_signal != 1:
                continue

            if ticker not in tomorrow_lookup:
                continue

            exec_row = tomorrow_lookup[ticker]

            entry_price = position['entry_price']
            exit_price = float(exec_row['aO'])
            qty = position['qty']

            pnl = _long_pnl(
                entry_price,
                exit_price,
                qty
            )

            exit_fee = (
                exit_price *
                qty *
                BROKERAGE_RATE
            )

            # Entry cost was already removed at entry.
            balance += (
                exit_price * qty -
                exit_fee
            )

            cumulative_pnl += pnl

            realized_equity = (
                INITIAL_BALANCE +
                cumulative_pnl
            )

            trade_history.append(
                _trade_record(
                    trade_id_counter,
                    position['entry_time'],
                    exec_row['dt'],
                    ticker,
                    'LONG',
                    pnl,
                    realized_equity
                )
            )

            trade_id_counter += 1
            tickers_to_remove.append(ticker)

        for ticker in tickers_to_remove:
            del active_positions[ticker]

        # ========================================================
        # B. PORTFOLIO VALUE AT TOMORROW'S OPEN
        # ========================================================

        execution_portfolio_value = _portfolio_value(
            tomorrow_data,
            active_positions,
            balance
        )

        # ========================================================
        # C. ORDINARY LONG ENTRIES
        # ========================================================

        if 'Conviction_Score' not in today_data.columns:
            today_data['Conviction_Score'] = 0.0

        today_data = today_data.sort_values(
            by='Conviction_Score',
            ascending=False
        )

        for _, row in today_data.iterrows():

            ticker = row['tkr']

            # Pair trades are handled separately.
            if row.get('Pair_DayTrade', 0) == 1:
                continue

            if ticker in active_positions:
                continue

            if row.get('Buy_Signal', 0) != 1:
                continue

            if ticker not in tomorrow_lookup:
                continue

            exec_row = tomorrow_lookup[ticker]

            entry_price = float(exec_row['aO'])

            # CRITICAL: use today's volume, not tomorrow's.
            prior_day_volume = float(row['aV'])

            qty = RiskAndSizingEngine().calculate_quantity(
                portfolio_value=execution_portfolio_value,
                available_cash=balance,
                entry_price=entry_price,
                prior_day_volume=prior_day_volume
            )

            if qty <= 0:
                continue

            entry_cost = (
                qty *
                entry_price
            )

            entry_fee = (
                entry_cost *
                BROKERAGE_RATE
            )

            total_outflow = (
                entry_cost +
                entry_fee
            )

            if balance < total_outflow:
                continue

            balance -= total_outflow

            active_positions[ticker] = {
                'qty': qty,
                'entry_price': entry_price,
                'entry_time': exec_row['dt']
            }

            execution_portfolio_value = _portfolio_value(
                tomorrow_data,
                active_positions,
                balance
            )

        # ========================================================
        # D. SHORT TRADES
        # ========================================================

        short_mask = today_data.get(
            'Short_Signal',
            pd.Series(
                0,
                index=today_data.index
            )
        ) == 1

        short_candidates = today_data[
            short_mask
        ].copy()

        if not short_candidates.empty:

            if 'Short_Conviction_Score' in short_candidates.columns:
                short_candidates = short_candidates.sort_values(
                    'Short_Conviction_Score',
                    ascending=False
                )

            for _, row in short_candidates.iterrows():

                ticker = row['tkr']

                if ticker in active_positions:
                    continue

                if ticker not in tomorrow_lookup:
                    continue

                exec_row = tomorrow_lookup[ticker]

                entry_price = float(exec_row['aO'])
                exit_price = float(exec_row['aC'])

                # Use signal-day volume.
                prior_day_volume = float(row['aV'])

                qty = RiskAndSizingEngine().calculate_quantity(
                    portfolio_value=execution_portfolio_value,
                    available_cash=balance,
                    entry_price=entry_price,
                    prior_day_volume=prior_day_volume
                )

                if qty <= 0:
                    continue

                pnl = _short_pnl(
                    entry_price,
                    exit_price,
                    qty
                )

                # Shorts are closed on the same day, so only the
                # realized P&L changes cash in this simplified model.
                balance += pnl
                cumulative_pnl += pnl

                realized_equity = (
                    INITIAL_BALANCE +
                    cumulative_pnl
                )

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        exec_row['dt'],
                        exec_row['dt'],
                        ticker,
                        'SHORT',
                        pnl,
                        realized_equity
                    )
                )

                trade_id_counter += 1

        # ========================================================
        # E. PAIRS TRADING
        # ========================================================

        pair_signals = today_data[
            today_data.get(
                'Pair_DayTrade',
                pd.Series(
                    0,
                    index=today_data.index
                )
            ) == 1
        ].copy()

        if not pair_signals.empty:

            for pair_id, pair_rows in pair_signals.groupby(
                'Pair_ID'
            ):

                if not pair_id or len(pair_rows) != 2:
                    continue

                long_rows = pair_rows[
                    pair_rows.get(
                        'Pair_Long_Signal',
                        0
                    ) == 1
                ]

                short_rows = pair_rows[
                    pair_rows.get(
                        'Pair_Short_Signal',
                        0
                    ) == 1
                ]

                if (
                    len(long_rows) != 1 or
                    len(short_rows) != 1
                ):
                    continue

                long_ticker = long_rows.iloc[0]['tkr']
                short_ticker = short_rows.iloc[0]['tkr']

                if (
                    long_ticker not in tomorrow_lookup or
                    short_ticker not in tomorrow_lookup
                ):
                    continue

                long_exec = tomorrow_lookup[long_ticker]
                short_exec = tomorrow_lookup[short_ticker]

                long_entry = float(long_exec['aO'])
                short_entry = float(short_exec['aO'])

                long_exit = float(long_exec['aC'])
                short_exit = float(short_exec['aC'])

                pair_budget = (
                    execution_portfolio_value *
                    MAX_PORTFOLIO_ALLOCATION
                )

                long_prior_volume = float(
                    long_rows.iloc[0]['aV']
                )

                short_prior_volume = float(
                    short_rows.iloc[0]['aV']
                )

                long_qty = RiskAndSizingEngine().calculate_quantity(
                    portfolio_value=execution_portfolio_value,
                    available_cash=balance,
                    entry_price=long_entry,
                    prior_day_volume=long_prior_volume
                )

                short_qty = RiskAndSizingEngine().calculate_quantity(
                    portfolio_value=execution_portfolio_value,
                    available_cash=balance,
                    entry_price=short_entry,
                    prior_day_volume=short_prior_volume
                )

                if long_qty <= 0 or short_qty <= 0:
                    continue

                # Equal-notional pair exposure.
                target_notional = min(
                    long_qty * long_entry,
                    short_qty * short_entry,
                    pair_budget / 2
                )

                long_qty = int(
                    target_notional /
                    long_entry
                )

                short_qty = int(
                    target_notional /
                    short_entry
                )

                if long_qty <= 0 or short_qty <= 0:
                    continue

                long_pnl = _long_pnl(
                    long_entry,
                    long_exit,
                    long_qty
                )

                short_pnl = _short_pnl(
                    short_entry,
                    short_exit,
                    short_qty
                )

                pair_pnl = (
                    long_pnl +
                    short_pnl
                )

                cumulative_pnl += pair_pnl
                balance += pair_pnl

                realized_equity = (
                    INITIAL_BALANCE +
                    cumulative_pnl
                )

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        long_exec['dt'],
                        long_exec['dt'],
                        long_ticker,
                        'PAIR_LONG',
                        long_pnl,
                        realized_equity,
                        pair_id
                    )
                )

                trade_id_counter += 1

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        short_exec['dt'],
                        short_exec['dt'],
                        short_ticker,
                        'PAIR_SHORT',
                        short_pnl,
                        realized_equity,
                        pair_id
                    )
                )

                trade_id_counter += 1

    # ============================================================
    # FORCE-CLOSE REMAINING LONG POSITIONS
    # ============================================================

    if len(unique_dates) > 0:

        final_date = unique_dates[-1]
        final_data = df[
            df['dt'] == final_date
        ]

        for ticker, position in list(
            active_positions.items()
        ):

            final_row = final_data[
                final_data['tkr'] == ticker
            ]

            if final_row.empty:
                continue

            exec_row = final_row.iloc[0]

            entry_price = position['entry_price']
            exit_price = float(exec_row['aC'])
            qty = position['qty']

            pnl = _long_pnl(
                entry_price,
                exit_price,
                qty
            )

            exit_fee = (
                exit_price *
                qty *
                BROKERAGE_RATE
            )

            balance += (
                exit_price * qty -
                exit_fee
            )

            cumulative_pnl += pnl

            realized_equity = (
                INITIAL_BALANCE +
                cumulative_pnl
            )

            trade_history.append(
                _trade_record(
                    trade_id_counter,
                    position['entry_time'],
                    exec_row['dt'],
                    ticker,
                    'LONG',
                    pnl,
                    realized_equity
                )
            )

            trade_id_counter += 1

            del active_positions[ticker]

    return pd.DataFrame(trade_history)
