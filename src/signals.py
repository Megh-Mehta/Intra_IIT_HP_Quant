import talib
import pandas as pd
import numpy as np


# ============================================================
# 1. EXISTING LONG STRATEGY: EMA CROSSOVER
# ============================================================

def strat_long_ema_crossover(group: pd.DataFrame) -> pd.DataFrame:
    """Long Strategy: Hyper-Fast 3/8 EMA Crossover."""
    group = group.copy()

    group['EMA_3'] = talib.EMA(group['aC'], timeperiod=3)
    group['EMA_8'] = talib.EMA(group['aC'], timeperiod=8)

    group['Buy_Signal'] = np.where(
        (group['EMA_3'] > group['EMA_8']) &
        (group['EMA_3'].shift(1) <= group['EMA_8'].shift(1)),
        1, 0
    )

    group['Exit_Signal'] = np.where(
        (group['EMA_3'] < group['EMA_8']) &
        (group['EMA_3'].shift(1) >= group['EMA_8'].shift(1)),
        1, 0
    )

    spread = ((group['EMA_3'] - group['EMA_8']) / group['EMA_8']) * 100
    group['Conviction_Score'] = np.where(
        group['Buy_Signal'] == 1, spread, 0
    )

    return group


# ============================================================
# 2. EXISTING LONG STRATEGY: BOLLINGER + RSI
# ============================================================

def strat_long_bollinger(group: pd.DataFrame) -> pd.DataFrame:
    """Long Strategy: Bollinger Bands + RSI Mean Reversion."""
    group = group.copy()

    group['upper'], group['middle'], group['lower'] = talib.BBANDS(
        group['aC'],
        timeperiod=10,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )

    group['RSI'] = talib.RSI(group['aC'], timeperiod=10)

    group['Buy_Signal'] = np.where(
        (group['aC'] < group['lower']) &
        (group['RSI'] < 40),
        1, 0
    )

    group['Exit_Signal'] = np.where(
        group['aC'] > group['middle'],
        1, 0
    )

    crash_depth = ((group['lower'] - group['aC']) / group['lower']) * 100

    group['Conviction_Score'] = np.where(
        group['Buy_Signal'] == 1,
        crash_depth,
        0
    )

    return group


# ============================================================
# 3. NEW LONG STRATEGY: STATISTICAL MEAN REVERSION
# ============================================================

def strat_long_mean_reversion(group: pd.DataFrame) -> pd.DataFrame:
    """
    Long Mean Reversion:
    Buy unusually depressed prices and exit when they revert
    toward the rolling mean.

    Uses:
      - 20-day rolling mean
      - 20-day rolling standard deviation
      - Z-score
      - RSI confirmation
    """
    group = group.copy()

    window = 20

    group['MR_Mean'] = group['aC'].rolling(window).mean()
    group['MR_Std'] = group['aC'].rolling(window).std()
    group['MR_Z'] = (
        (group['aC'] - group['MR_Mean']) /
        group['MR_Std'].replace(0, np.nan)
    )
    group['MR_RSI'] = talib.RSI(group['aC'], timeperiod=10)

    # Strong negative deviation + oversold confirmation.
    group['Buy_Signal'] = np.where(
        (group['MR_Z'] < -2.0) &
        (group['MR_RSI'] < 35),
        1, 0
    )

    # Exit after reversion toward the mean.
    group['Exit_Signal'] = np.where(
        group['MR_Z'] > -0.25,
        1, 0
    )

    group['Conviction_Score'] = np.where(
        group['Buy_Signal'] == 1,
        np.abs(group['MR_Z']),
        0
    )

    return group


# ============================================================
# 4. NEW SHORT STRATEGY: SHORT MEAN REVERSION
# ============================================================

def strat_short_mean_reversion(group: pd.DataFrame) -> pd.DataFrame:
    """
    Short Mean Reversion:
    Short unusually extended upward moves.

    IMPORTANT:
    The engine closes this short on the same trading day it
    enters, because overnight shorts are not permitted.
    """
    group = group.copy()

    window = 20

    group['MR_Mean'] = group['aC'].rolling(window).mean()
    group['MR_Std'] = group['aC'].rolling(window).std()
    group['MR_Z'] = (
        (group['aC'] - group['MR_Mean']) /
        group['MR_Std'].replace(0, np.nan)
    )
    group['MR_RSI'] = talib.RSI(group['aC'], timeperiod=10)

    # Price is statistically stretched upward.
    group['Short_Signal'] = np.where(
        (group['MR_Z'] > 2.0) &
        (group['MR_RSI'] > 65),
        1, 0
    )

    group['Short_Conviction_Score'] = np.where(
        group['Short_Signal'] == 1,
        group['MR_Z'],
        0
    )

    return group


# ============================================================
# 5. NEW SHORT STRATEGY: MOMENTUM BREAKDOWN
# ============================================================

def strat_short_breakdown(group: pd.DataFrame) -> pd.DataFrame:
    """
    Short Momentum / Breakdown:
    Looks for a downside break supported by negative momentum
    and unusually high volume.

    Shorts are always closed the same day by the engine.
    """
    group = group.copy()

    group['EMA_5'] = talib.EMA(group['aC'], timeperiod=5)
    group['EMA_15'] = talib.EMA(group['aC'], timeperiod=15)
    group['RSI'] = talib.RSI(group['aC'], timeperiod=10)

    group['Volume_Mean'] = group['aV'].rolling(20).mean()

    # Today's close breaks below yesterday's recent low.
    group['Breakdown'] = (
        group['aC'] <
        group['aC'].shift(1).rolling(5).min()
    )

    group['Short_Signal'] = np.where(
        group['Breakdown'] &
        (group['EMA_5'] < group['EMA_15']) &
        (group['RSI'] < 45) &
        (group['aV'] > group['Volume_Mean'] * 1.25),
        1, 0
    )

    # Stronger downside momentum = higher conviction.
    downside_strength = (
        (group['EMA_15'] - group['EMA_5']) /
        group['EMA_15'].replace(0, np.nan)
    ) * 100

    group['Short_Conviction_Score'] = np.where(
        group['Short_Signal'] == 1,
        downside_strength,
        0
    )

    return group


# ============================================================
# 6. PAIRS TRADING
# ============================================================

def strat_pairs_trading(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling pairs trading strategy.

    Pair selection:
      - Uses only historical data available before each signal.
      - Finds the most correlated pair using 30-day returns.

    Spread:
      log(P_A) - beta * log(P_B)

    Signal:
      z > +2  -> short A / long B
      z < -2  -> long A / short B

    Because the competition does not permit overnight shorts,
    the pair is treated as a T+1-open -> same-day-close trade.

    The engine recognizes:
      Pair_Long_Signal
      Pair_Short_Signal
      Pair_DayTrade = 1
    """
    df = df.copy()
    df = df.sort_values(['dt', 'tkr']).reset_index(drop=True)

    df['Pair_Long_Signal'] = 0
    df['Pair_Short_Signal'] = 0
    df['Pair_DayTrade'] = 0
    df['Pair_Conviction_Score'] = 0.0
    df['Pair_ID'] = ''

    tickers = sorted(df['tkr'].dropna().unique())

    if len(tickers) < 2:
        return df

    price_wide = df.pivot(index='dt', columns='tkr', values='aC').sort_index()
    returns = price_wide.pct_change()

    dates = price_wide.index

    for i in range(30, len(dates)):
        current_date = dates[i]

        # Historical data only: no future information.
        hist_returns = returns.iloc[max(0, i - 30):i]

        valid_tickers = [
            t for t in tickers
            if hist_returns[t].notna().sum() >= 20
        ]

        if len(valid_tickers) < 2:
            continue

        corr = hist_returns[valid_tickers].corr()

        best_pair = None
        best_corr = -np.inf

        for a_idx in range(len(valid_tickers)):
            for b_idx in range(a_idx + 1, len(valid_tickers)):
                a = valid_tickers[a_idx]
                b = valid_tickers[b_idx]
                c = corr.loc[a, b]

                if pd.notna(c) and c > best_corr:
                    best_corr = c
                    best_pair = (a, b)

        if best_pair is None or best_corr < 0.70:
            continue

        a, b = best_pair

        # Historical price relationship.
        hist_prices = price_wide.iloc[max(0, i - 30):i][[a, b]].dropna()

        if len(hist_prices) < 20:
            continue

        log_a = np.log(hist_prices[a])
        log_b = np.log(hist_prices[b])

        variance_b = np.var(log_b)

        if variance_b == 0:
            continue

        beta = np.cov(log_a, log_b, ddof=0)[0, 1] / variance_b

        historical_spread = log_a - beta * log_b

        spread_mean = historical_spread.mean()
        spread_std = historical_spread.std()

        if spread_std == 0 or pd.isna(spread_std):
            continue

        current_a = price_wide.loc[current_date, a]
        current_b = price_wide.loc[current_date, b]

        if pd.isna(current_a) or pd.isna(current_b):
            continue

        current_spread = np.log(current_a) - beta * np.log(current_b)
        z = (current_spread - spread_mean) / spread_std

        pair_id = f'{a}_{b}'

        # Spread too high:
        # short A, long B.
        if z > 2.0:
            mask_a = (df['dt'] == current_date) & (df['tkr'] == a)
            mask_b = (df['dt'] == current_date) & (df['tkr'] == b)

            df.loc[mask_a, 'Pair_Short_Signal'] = 1
            df.loc[mask_b, 'Pair_Long_Signal'] = 1

            df.loc[mask_a, 'Pair_DayTrade'] = 1
            df.loc[mask_b, 'Pair_DayTrade'] = 1

            df.loc[mask_a, 'Pair_Conviction_Score'] = abs(z)
            df.loc[mask_b, 'Pair_Conviction_Score'] = abs(z)

            df.loc[mask_a, 'Pair_ID'] = pair_id
            df.loc[mask_b, 'Pair_ID'] = pair_id

        # Spread too low:
        # long A, short B.
        elif z < -2.0:
            mask_a = (df['dt'] == current_date) & (df['tkr'] == a)
            mask_b = (df['dt'] == current_date) & (df['tkr'] == b)

            df.loc[mask_a, 'Pair_Long_Signal'] = 1
            df.loc[mask_b, 'Pair_Short_Signal'] = 1

            df.loc[mask_a, 'Pair_DayTrade'] = 1
            df.loc[mask_b, 'Pair_DayTrade'] = 1

            df.loc[mask_a, 'Pair_Conviction_Score'] = abs(z)
            df.loc[mask_b, 'Pair_Conviction_Score'] = abs(z)

            df.loc[mask_a, 'Pair_ID'] = pair_id
            df.loc[mask_b, 'Pair_ID'] = pair_id

    return df


# This flag tells the engine that this strategy needs the entire
# cross-section instead of one ticker at a time.
strat_pairs_trading.requires_full_data = True


# ============================================================
# 7. CAUSAL BOLLINGER + WILDER RSI
# ============================================================

BB_LOOKBACK = 15
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 40.0
RSI_OVERBOUGHT = 60.0


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder-style RSI using only current and historical closes."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)

    return 100.0 - (100.0 / (1.0 + rs))


def strat_causal_bollinger_rsi(group: pd.DataFrame) -> pd.DataFrame:
    """
    Causal Bollinger/RSI mean-reversion strategy.

    A signal on row i uses only data through row i.
    The execution engine acts on that signal at row i+1 Open.

    Long:
        Enter below the lower Bollinger Band with RSI < 40.
        Continue holding until price reaches the middle band.

    Short:
        Enter above the upper Bollinger Band with RSI > 60.
        The engine closes the short at the same day's Close.
    """
    out = group.copy()

    close = out['aC'].astype(float)

    middle = close.rolling(
        BB_LOOKBACK,
        min_periods=BB_LOOKBACK
    ).mean()

    std = close.rolling(
        BB_LOOKBACK,
        min_periods=BB_LOOKBACK
    ).std(ddof=0)

    upper = middle + BB_STD * std
    lower = middle - BB_STD * std
    rsi = wilder_rsi(close, RSI_PERIOD)

    signal = np.zeros(len(out), dtype=np.int8)
    holding_long = False

    for i in range(len(out)):
        if pd.isna(middle.iloc[i]) or pd.isna(rsi.iloc[i]):
            continue

        price = close.iloc[i]
        mid = middle.iloc[i]
        up = upper.iloc[i]
        low = lower.iloc[i]
        rsi_value = rsi.iloc[i]

        if holding_long:
            if price >= mid:
                holding_long = False
            else:
                signal[i] = 1
                continue

        if price < low and rsi_value < RSI_OVERSOLD:
            holding_long = True
            signal[i] = 1

        elif price > up and rsi_value > RSI_OVERBOUGHT:
            signal[i] = -1

    out['raw_signal'] = signal

    # Translate the uploaded strategy's state signal into the
    # repository's common engine interface.
    out['Buy_Signal'] = (
        (out['raw_signal'] == 1) &
        (out['raw_signal'].shift(1).fillna(0) != 1)
    ).astype(int)

    out['Exit_Signal'] = (
        (out['raw_signal'].shift(1).fillna(0) == 1) &
        (out['raw_signal'] != 1)
    ).astype(int)

    out['Short_Signal'] = (
        out['raw_signal'] == -1
    ).astype(int)

    out['Conviction_Score'] = np.where(
        out['raw_signal'] != 0,
        np.abs(
            (close - middle) /
            (std.replace(0, np.nan))
        ),
        0.0
    )

    return out


# ============================================================
# STRATEGY REGISTRY
# ============================================================

strategies = {
    "EMA_Crossover": strat_long_ema_crossover,
    "Bollinger_Bands": strat_long_bollinger,
    "Mean_Reversion": strat_long_mean_reversion,
    "Pairs_Trading": strat_pairs_trading,
    "Short_Mean_Reversion": strat_short_mean_reversion,
    "Short_Breakdown": strat_short_breakdown,

    "Causal_Bollinger_RSI": strat_causal_bollinger_rsi,
}
