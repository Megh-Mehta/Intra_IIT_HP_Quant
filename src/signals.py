import talib
import pandas as pd
import numpy as np


# ============================================================
# 1. COMBINED LONG + SHORT MEAN REVERSION
# ============================================================

def strat_mean_reversion(group: pd.DataFrame) -> pd.DataFrame:
    """
    Combined Mean Reversion strategy.

    Long:
      - Buy when price is more than 2 standard deviations below
        the 20-day rolling mean.
      - Require RSI < 35.
      - Exit when the z-score reverts above -0.25.

    Short:
      - Short when price is more than 2 standard deviations above
        the 20-day rolling mean.
      - Require RSI > 65.
      - The execution engine closes the short on the same
        trading day because overnight shorts are not permitted.

    Both directions are emitted by the same strategy and therefore
    appear together in one trade CSV.
    """
    group = group.copy()

    window = 20

    group['MR_Mean'] = group['aC'].rolling(window).mean()
    group['MR_Std'] = group['aC'].rolling(window).std()
    group['MR_Z'] = (
        (group['aC'] - group['MR_Mean']) /
        group['MR_Std'].replace(0, np.nan)
    )

    group['MR_RSI'] = talib.RSI(
        group['aC'],
        timeperiod=10
    )

    # ------------------------------------------------------------
    # LONG SIDE
    # ------------------------------------------------------------

    group['Buy_Signal'] = np.where(
        (group['MR_Z'] < -2.0) &
        (group['MR_RSI'] < 35),
        1,
        0
    )

    group['Exit_Signal'] = np.where(
        group['MR_Z'] > -0.25,
        1,
        0
    )

    # ------------------------------------------------------------
    # SHORT SIDE
    # ------------------------------------------------------------

    group['Short_Signal'] = np.where(
        (group['MR_Z'] > 2.0) &
        (group['MR_RSI'] > 65),
        1,
        0
    )

    # Use the same conviction concept for both directions.
    group['Conviction_Score'] = np.where(
        (group['Buy_Signal'] == 1) |
        (group['Short_Signal'] == 1),
        np.abs(group['MR_Z']),
        0
    )

    group['Short_Conviction_Score'] = np.where(
        group['Short_Signal'] == 1,
        np.abs(group['MR_Z']),
        0
    )

    return group


# ============================================================
# 2. CAUSAL BOLLINGER + WILDER RSI
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
            std.replace(0, np.nan)
        ),
        0.0
    )

    out['Short_Conviction_Score'] = np.where(
        out['Short_Signal'] == 1,
        out['Conviction_Score'],
        0.0
    )

    return out


# ============================================================
# STRATEGY REGISTRY
# ============================================================

strategies = {
    "Mean_Reversion": strat_mean_reversion,
    "Causal_Bollinger_RSI": strat_causal_bollinger_rsi,
}
