import pandas as pd
import numpy as np


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
        'P&L': round(pnl, 2),
        'Running_Balance': round(running_balance, 2)
    }


def _quantity(balance, entry_price, volume, max_portfolio_pct,
              max_volume_pct, brokerage_rate):
    if pd.isna(entry_price) or entry_price <= 0:
        return 0

    max_invest = balance * max_portfolio_pct

    qty_balance = int(
        max_invest / (entry_price * (1 + brokerage_rate))
    )

    qty_volume = int(max(volume, 0) * max_volume_pct)

    return max(0, min(qty_balance, qty_volume))


def _long_pnl(entry_price, exit_price, qty, brokerage_rate):
    entry_cost = entry_price * qty * (1 + brokerage_rate)
    net_exit = exit_price * qty * (1 - brokerage_rate)
    return net_exit - entry_cost


def _short_pnl(entry_price, exit_price, qty, brokerage_rate):
    """
    Short P&L:
        Sell at entry -> buy back at exit.

    Brokerage is charged on both legs.
    """
    short_proceeds = entry_price * qty * (1 - brokerage_rate)
    cover_cost = exit_price * qty * (1 + brokerage_rate)
    return short_proceeds - cover_cost


def _prepare_strategy_data(df_raw, strategy_func):
    """
    Normal ticker-by-ticker strategies receive one ticker at a time.

    Pairs trading needs the entire cross-section to identify the pair,
    so it is marked with requires_full_data=True in signals.py.
    """
    if getattr(strategy_func, 'requires_full_data', False):
        return strategy_func(
            df_raw.sort_values(['dt', 'tkr']).copy()
        )

    processed_dfs = []

    for _, group_df in df_raw.groupby('tkr'):
        processed_dfs.append(strategy_func(group_df.copy()))

    return pd.concat(processed_dfs, ignore_index=True)


def run_backtest(df_raw: pd.DataFrame, strategy_func) -> pd.DataFrame:
    """
    Rule-compliant T+1 backtest.

    Long:
        Signal on day i -> enter at day i+1 Open.
        Can remain open overnight.

    Short:
        Signal on day i -> enter at day i+1 Open.
        Must be closed at day i+1 Close.

    Pairs:
        Pair signal on day i -> both legs enter at day i+1 Open.
        The short leg is closed at day i+1 Close, so the pair trade
        is intraday on the execution day.
    """
    df = _prepare_strategy_data(df_raw, strategy_func)
    df = df.sort_values(['dt', 'tkr']).reset_index(drop=True)

    initial_capital = 1_000_000.0
    balance = initial_capital
    cumulative_pnl = 0.0

    brokerage_rate = 0.0005
    max_portfolio_pct = 0.15
    max_volume_pct = 0.10
    trailing_stop_pct = 0.05

    long_positions = {}
    trade_history = []
    trade_id_counter = 1

    unique_dates = df['dt'].sort_values().unique()

    # ------------------------------------------------------------
    # T+1 execution loop
    # ------------------------------------------------------------
    for i in range(len(unique_dates) - 1):

        today_date = unique_dates[i]
        tomorrow_date = unique_dates[i + 1]

        today_data = df[df['dt'] == today_date].copy()
        tomorrow_data = df[df['dt'] == tomorrow_date].copy()

        tomorrow_lookup = {
            row['tkr']: row
            for _, row in tomorrow_data.iterrows()
        }

        # ========================================================
        # A. EXIT EXISTING LONG POSITIONS
        # ========================================================

        tickers_to_remove = []

        for tkr, pos in long_positions.items():

            today_ticker = today_data[
                today_data['tkr'] == tkr
            ]

            if today_ticker.empty:
                continue

            row = today_ticker.iloc[0]

            # Update trailing high.
            if row['aH'] > pos['highest_aH']:
                pos['highest_aH'] = row['aH']

            stop_price = (
                pos['highest_aH'] *
                (1 - trailing_stop_pct)
            )

            exit_signal = row.get('Exit_Signal', 0)

            if row['aC'] < stop_price or exit_signal == 1:

                if tkr not in tomorrow_lookup:
                    continue

                exec_row = tomorrow_lookup[tkr]

                exit_price = exec_row['aO']
                qty = pos['qty']

                pnl = _long_pnl(
                    pos['entry_price'],
                    exit_price,
                    qty,
                    brokerage_rate
                )

                net_exit = (
                    exit_price *
                    qty *
                    (1 - brokerage_rate)
                )

                balance += net_exit
                cumulative_pnl += pnl

                realized_equity = (
                    initial_capital +
                    cumulative_pnl
                )

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        pos['entry_time'],
                        exec_row['dt'],
                        tkr,
                        'LONG',
                        pnl,
                        realized_equity
                    )
                )

                trade_id_counter += 1
                tickers_to_remove.append(tkr)

        for tkr in tickers_to_remove:
            del long_positions[tkr]

        # ========================================================
        # B. RANK ORDINARY LONG SIGNALS
        # ========================================================

        if 'Conviction_Score' not in today_data.columns:
            today_data['Conviction_Score'] = 0.0

        today_data = today_data.sort_values(
            by='Conviction_Score',
            ascending=False
        )

        # ========================================================
        # C. ORDINARY LONG ENTRIES
        # ========================================================

        for _, row in today_data.iterrows():

            tkr = row['tkr']

            # Pair trades are handled separately below.
            if row.get('Pair_DayTrade', 0) == 1:
                continue

            if tkr in long_positions:
                continue

            if row.get('Buy_Signal', 0) != 1:
                continue

            if tkr not in tomorrow_lookup:
                continue

            exec_row = tomorrow_lookup[tkr]

            entry_price = exec_row['aO']

            qty = _quantity(
                balance,
                entry_price,
                exec_row['aV'],
                max_portfolio_pct,
                max_volume_pct,
                brokerage_rate
            )

            if qty <= 0:
                continue

            total_deduction = (
                qty *
                entry_price *
                (1 + brokerage_rate)
            )

            if balance < total_deduction:
                continue

            balance -= total_deduction

            long_positions[tkr] = {
                'qty': qty,
                'entry_price': entry_price,
                'highest_aH': entry_price,
                'entry_time': exec_row['dt']
            }

        # ========================================================
        # D. SHORT STRATEGIES
        #
        # Signal today -> short at tomorrow Open -> cover at
        # tomorrow Close. This prevents overnight short exposure.
        # ========================================================

        short_candidates = today_data[
            today_data.get(
                'Short_Signal',
                pd.Series(0, index=today_data.index)
            ) == 1
        ].copy()

        if not short_candidates.empty:

            if 'Short_Conviction_Score' in short_candidates.columns:
                short_candidates = short_candidates.sort_values(
                    'Short_Conviction_Score',
                    ascending=False
                )

            for _, row in short_candidates.iterrows():

                tkr = row['tkr']

                if tkr in long_positions:
                    continue

                if tkr not in tomorrow_lookup:
                    continue

                exec_row = tomorrow_lookup[tkr]

                entry_price = exec_row['aO']
                exit_price = exec_row['aC']

                qty = _quantity(
                    balance,
                    entry_price,
                    exec_row['aV'],
                    max_portfolio_pct,
                    max_volume_pct,
                    brokerage_rate
                )

                if qty <= 0:
                    continue

                pnl = _short_pnl(
                    entry_price,
                    exit_price,
                    qty,
                    brokerage_rate
                )

                # Short positions don't consume the same cash flow
                # as a long purchase in this simplified competition
                # accounting model. Reserve the notional as margin.
                margin = (
                    entry_price *
                    qty *
                    (1 + brokerage_rate)
                )

                if margin > balance:
                    continue

                balance += pnl
                cumulative_pnl += pnl

                realized_equity = (
                    initial_capital +
                    cumulative_pnl
                )

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        exec_row['dt'],
                        exec_row['dt'],
                        tkr,
                        'SHORT',
                        pnl,
                        realized_equity
                    )
                )

                trade_id_counter += 1

        # ========================================================
        # E. PAIRS TRADING
        #
        # The pair signal exists on today. Both legs execute at
        # tomorrow's open. Both legs are closed at tomorrow's close.
        # ========================================================

        pair_signals = today_data[
            today_data.get(
                'Pair_DayTrade',
                pd.Series(0, index=today_data.index)
            ) == 1
        ].copy()

        if not pair_signals.empty:

            for pair_id, pair_rows in pair_signals.groupby('Pair_ID'):

                if not pair_id or len(pair_rows) != 2:
                    continue

                # We require one long and one short leg.
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

                if len(long_rows) != 1 or len(short_rows) != 1:
                    continue

                long_tkr = long_rows.iloc[0]['tkr']
                short_tkr = short_rows.iloc[0]['tkr']

                if (
                    long_tkr not in tomorrow_lookup or
                    short_tkr not in tomorrow_lookup
                ):
                    continue

                long_exec = tomorrow_lookup[long_tkr]
                short_exec = tomorrow_lookup[short_tkr]

                long_entry = long_exec['aO']
                short_entry = short_exec['aO']

                long_exit = long_exec['aC']
                short_exit = short_exec['aC']

                # Use equal notional for the two legs.
                pair_budget = (
                    balance *
                    max_portfolio_pct
                )

                long_qty = _quantity(
                    balance,
                    long_entry,
                    long_exec['aV'],
                    max_portfolio_pct,
                    max_volume_pct,
                    brokerage_rate
                )

                short_qty = _quantity(
                    balance,
                    short_entry,
                    short_exec['aV'],
                    max_portfolio_pct,
                    max_volume_pct,
                    brokerage_rate
                )

                if long_qty <= 0 or short_qty <= 0:
                    continue

                # Adjust to approximately equal dollar exposure.
                target_notional = min(
                    long_qty * long_entry,
                    short_qty * short_entry,
                    pair_budget / 2
                )

                long_qty = int(target_notional / long_entry)
                short_qty = int(target_notional / short_entry)

                if long_qty <= 0 or short_qty <= 0:
                    continue

                long_pnl = _long_pnl(
                    long_entry,
                    long_exit,
                    long_qty,
                    brokerage_rate
                )

                short_pnl = _short_pnl(
                    short_entry,
                    short_exit,
                    short_qty,
                    brokerage_rate
                )

                pair_pnl = long_pnl + short_pnl

                cumulative_pnl += pair_pnl
                balance += pair_pnl

                realized_equity = (
                    initial_capital +
                    cumulative_pnl
                )

                trade_history.append(
                    _trade_record(
                        trade_id_counter,
                        long_exec['dt'],
                        long_exec['dt'],
                        long_tkr,
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
                        short_tkr,
                        'PAIR_SHORT',
                        short_pnl,
                        realized_equity,
                        pair_id
                    )
                )

                trade_id_counter += 1

    # ============================================================
    # FORCE CLOSE REMAINING LONG POSITIONS
    # ============================================================

    if len(unique_dates) > 0:

        last_date = unique_dates[-1]
        last_data = df[df['dt'] == last_date]

        for tkr, pos in long_positions.items():

            last_ticker = last_data[
                last_data['tkr'] == tkr
            ]

            if last_ticker.empty:
                continue

            exec_row = last_ticker.iloc[0]

            qty = pos['qty']

            pnl = _long_pnl(
                pos['entry_price'],
                exec_row['aC'],
                qty,
                brokerage_rate
            )

            net_exit = (
                exec_row['aC'] *
                qty *
                (1 - brokerage_rate)
            )

            balance += net_exit
            cumulative_pnl += pnl

            realized_equity = (
                initial_capital +
                cumulative_pnl
            )

            trade_history.append(
                _trade_record(
                    trade_id_counter,
                    pos['entry_time'],
                    exec_row['dt'],
                    tkr,
                    'LONG',
                    pnl,
                    realized_equity
                )
            )

            trade_id_counter += 1

    return pd.DataFrame(trade_history)
