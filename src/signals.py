import talib
import pandas as pd
import numpy as np


# ============================================================
# MEAN REVERSION COMPONENT
# ============================================================

def strat_mean_reversion(group: pd.DataFrame) -> pd.DataFrame:
    """
    Long + short statistical mean reversion component.

    Signals use only information through the current EOD row.
    Execution is performed by the engine on the next trading day's Open.

    LONG:
      Z-score < -2 and RSI < 35.
      Exit when Z-score > -0.25.

    SHORT:
      Z-score > +2 and RSI > 65.
      Shorts are closed at the same day's Close.
    """
    out = group.copy()
    window = 20

    out['MR_Mean'] = out['aC'].rolling(window).mean()
    out['MR_Std'] = out['aC'].rolling(window).std()
    out['MR_Z'] = (
        (out['aC'] - out['MR_Mean']) /
        out['MR_Std'].replace(0, np.nan)
    )

    out['MR_RSI'] = talib.RSI(
        out['aC'],
        timeperiod=10
    )

    out['Buy_Signal'] = (
        (out['MR_Z'] < -2.0) &
        (out['MR_RSI'] < 35)
    ).astype(int)

    out['Exit_Signal'] = (
        out['MR_Z'] > -0.25
    ).astype(int)

    out['Short_Signal'] = (
        (out['MR_Z'] > 2.0) &
        (out['MR_RSI'] > 65)
    ).astype(int)

    out['Conviction_Score'] = np.where(
        (out['Buy_Signal'] == 1) |
        (out['Short_Signal'] == 1),
        np.abs(out['MR_Z']),
        0.0
    )

    out['Short_Conviction_Score'] = np.where(
        out['Short_Signal'] == 1,
        np.abs(out['MR_Z']),
        0.0
    )

    return out


# ============================================================
# CAUSAL BOLLINGER + WILDER RSI COMPONENT
# ============================================================

BB_LOOKBACK = 15
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 40.0
RSI_OVERBOUGHT = 60.0


def wilder_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD
) -> pd.Series:
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

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def strat_causal_bollinger_rsi(
    group: pd.DataFrame
) -> pd.DataFrame:
    """
    Stateful causal Bollinger/RSI component.

    LONG:
      Price below lower Bollinger Band and RSI < 40.
      Hold until price reaches the middle band.

    SHORT:
      Price above upper Bollinger Band and RSI > 60.
      The engine closes the short at that day's Close.
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

    signal = np.zeros(
        len(out),
        dtype=np.int8
    )

    holding_long = False

    for i in range(len(out)):
        if (
            pd.isna(middle.iloc[i]) or
            pd.isna(rsi.iloc[i])
        ):
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

        if (
            price < low and
            rsi_value < RSI_OVERSOLD
        ):
            holding_long = True
            signal[i] = 1

        elif (
            price > up and
            rsi_value > RSI_OVERBOUGHT
        ):
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
# FINAL COMBINED STRATEGY
# ============================================================

def strat_final_strategy(
    group: pd.DataFrame
) -> pd.DataFrame:
    """
    Final ensemble strategy.

    The Mean Reversion and Causal Bollinger/RSI components are
    combined at the signal layer and then traded as ONE portfolio.

    If both components agree on direction, conviction is strengthened.
    If they disagree on the same row, no new position is opened.

    LONG:
      Mean Reversion OR Causal Bollinger/RSI long signal.

    SHORT:
      Mean Reversion OR Causal Bollinger/RSI short signal.

    A long is exited when either component produces its long-exit
    signal. Shorts remain same-day trades under the execution engine.
    """
    mean = strat_mean_reversion(group)
    causal = strat_causal_bollinger_rsi(group)

    out = group.copy()

    mean_long = mean['Buy_Signal'].astype(int)
    causal_long = causal['Buy_Signal'].astype(int)

    mean_short = mean['Short_Signal'].astype(int)
    causal_short = causal['Short_Signal'].astype(int)

    # Conflicting long/short signals are skipped rather than forcing
    # the portfolio to take opposite positions on the same ticker/day.
    long_signal = (
        ((mean_long == 1) | (causal_long == 1)) &
        ~((mean_short == 1) | (causal_short == 1))
    ).astype(int)

    short_signal = (
        ((mean_short == 1) | (causal_short == 1)) &
        ~((mean_long == 1) | (causal_long == 1))
    ).astype(int)

    out['Buy_Signal'] = long_signal

    out['Exit_Signal'] = (
        (mean['Exit_Signal'] == 1) |
        (causal['Exit_Signal'] == 1)
    ).astype(int)

    out['Short_Signal'] = short_signal

    out['Conviction_Score'] = np.maximum(
        mean['Conviction_Score'].fillna(0.0),
        causal['Conviction_Score'].fillna(0.0)
    )

    out['Short_Conviction_Score'] = np.maximum(
        mean['Short_Conviction_Score'].fillna(0.0),
        causal['Short_Conviction_Score'].fillna(0.0)
    )

    out['Mean_Reversion_Long'] = mean_long
    out['Mean_Reversion_Short'] = mean_short
    out['Causal_Bollinger_Long'] = causal_long
    out['Causal_Bollinger_Short'] = causal_short

    return out


# ============================================================
# STRATEGY REGISTRY
# ============================================================

strategies = {
    "final_strategy": strat_final_strategy,
}
