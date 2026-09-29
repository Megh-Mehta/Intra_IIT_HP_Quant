from pathlib import Path

import pandas as pd

from src.signals import strategies
from src.engine import run_backtest
from src.metrics import calculate_metrics


ROOT = Path(__file__).resolve().parent

TRAIN_DATA = ROOT / "data" / "hackathon_train_data.csv"
EVAL_DATA = ROOT / "hackathon_eval_secret.csv"
EVAL_DATA_FALLBACK = ROOT / "data" / "hackathon_eval_secret.csv"

FINAL_RESULTS = ROOT / "final_results.csv"
REPORT_FILE = ROOT / "FINAL_STRATEGY_REPORT.md"

REQUIRED_OUTPUT_COLUMNS = [
    "Trade_ID",
    "Entry_Time",
    "Exit_Time",
    "Ticker",
    "Direction",
    "P&L",
    "Running_Balance",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate one local EOD dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"dt", "tkr", "aO", "aH", "aL", "aC", "aV"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Dataset {path.name} is missing columns: {missing}"
        )

    df["dt"] = pd.to_datetime(df["dt"], utc=True)

    return df.sort_values(["tkr", "dt"]).reset_index(drop=True)


def clean_trade_output(trades: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the schema required by the rulebook."""
    if trades.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    output = trades.copy()

    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in output.columns:
            raise ValueError(
                f"Backtest output is missing required column: {column}"
            )

    output = output[REQUIRED_OUTPUT_COLUMNS].copy()
    output["Trade_ID"] = range(1, len(output) + 1)
    output["P&L"] = pd.to_numeric(output["P&L"], errors="raise").round(2)
    output["Running_Balance"] = pd.to_numeric(
        output["Running_Balance"], errors="raise"
    ).round(2)
    output["Direction"] = output["Direction"].astype(str)

    invalid_directions = set(output["Direction"]) - {"LONG", "SHORT"}
    if invalid_directions:
        raise ValueError(
            f"Invalid trade directions: {invalid_directions}"
        )

    return output


def build_training_report(
    training_trades: pd.DataFrame,
    metrics: dict,
) -> str:
    """Build the report from the actual training backtest."""
    total_pnl = (
        float(training_trades["P&L"].sum())
        if not training_trades.empty else 0.0
    )
    average_pnl = (
        float(training_trades["P&L"].mean())
        if not training_trades.empty else 0.0
    )
    gross_profit = (
        float(training_trades.loc[training_trades["P&L"] > 0, "P&L"].sum())
        if not training_trades.empty else 0.0
    )
    gross_loss = (
        float(training_trades.loc[training_trades["P&L"] < 0, "P&L"].sum())
        if not training_trades.empty else 0.0
    )

    long_trades = (
        int((training_trades["Direction"] == "LONG").sum())
        if not training_trades.empty else 0
    )
    short_trades = (
        int((training_trades["Direction"] == "SHORT").sum())
        if not training_trades.empty else 0
    )
    long_pnl = (
        float(training_trades.loc[
            training_trades["Direction"] == "LONG", "P&L"
        ].sum())
        if not training_trades.empty else 0.0
    )
    short_pnl = (
        float(training_trades.loc[
            training_trades["Direction"] == "SHORT", "P&L"
        ].sum())
        if not training_trades.empty else 0.0
    )

    return f"""# Final Strategy Report

## 1. Project Overview

This project implements an offline end-of-day algorithmic trading strategy
for the Intra IIT Tech Meet 2026 Day-Trading Quant Hackathon.

The program reads local market data, generates causal signals, executes
trades under the rulebook constraints, and produces the mandatory
final_results.csv file.

No external API or internet data is used during execution.

## 2. Final Strategy

The final strategy is named final_strategy.

It combines two mean-reversion components into one portfolio:

### Statistical Mean Reversion

A 20-day rolling mean and standard deviation are used to calculate a
price z-score.

Long:
- Z-score < -2
- RSI(10) < 35
- Exit when Z-score > -0.25

Short:
- Z-score > +2
- RSI(10) > 65
- Closed on the same day's Close

### Causal Bollinger + Wilder RSI

Parameters:
- 15-day Bollinger Band lookback
- 2 standard deviations
- 14-period Wilder RSI

Long:
- Price below the lower Bollinger Band
- RSI < 40
- Hold until price reaches the middle band

Short:
- Price above the upper Bollinger Band
- RSI > 60
- Closed on the same day's Close

## 3. Signal Combination

The two components are combined into one strategy and one portfolio.

If either component produces a long signal and neither produces a short
signal, final_strategy can enter LONG.

If either component produces a short signal and neither produces a long
signal, final_strategy can enter SHORT.

If the components disagree on direction on the same ticker/day, no new
position is opened. This avoids forcing an arbitrary choice between
contradictory signals.

The final conviction score is the larger component conviction score.

## 4. Why the Strategy Can Work

Both components express the hypothesis that prices can temporarily move
unusually far from a recent local equilibrium and subsequently partially
revert.

The statistical component measures abnormality with a rolling z-score and
uses RSI as confirmation.

The Bollinger/RSI component measures abnormality through Bollinger Band
distance and Wilder RSI.

Combining these related but distinct measurements can make the strategy
less dependent on one particular definition of price extension. The
disagreement filter also prevents an ambiguous signal from automatically
becoming a trade.

This is a statistical hypothesis, not a guarantee of future profitability.

## 5. Rulebook Compliance

The implementation follows the supplied rulebook:

- Initial balance: Rs 10,00,000.
- Entry brokerage: 0.05%.
- Exit brokerage: 0.05%.
- Trade value limited to 25% of current portfolio value.
- Quantity is a positive integer.
- Quantity limited to 10% of signal-day reported volume.
- Long positions may be carried.
- Short positions close the same trading day at that day's Close.
- Signal on day i executes at day i+1 Open.
- No external API calls are made.
- Signals use only current and historical observations.
- The hidden dataset is not used for strategy tuning.

## 6. Training-Data Results

These values are generated directly from the current training-data
backtest when main.py is executed.

| Metric | Training Result |
|---|---:|
| Trades | {metrics["Trades"]} |
| Net P/L | Rs {total_pnl:,.2f} |
| Net ROI | {metrics["Net ROI (%)"]:.2f}% |
| Sharpe Ratio | {metrics["Sharpe"]:.2f} |
| Maximum Drawdown | {metrics["Max DD (%)"]:.2f}% |
| Profit Factor | {metrics["Profit Factor"]:.2f} |
| Win Rate | {metrics["Win Rate (%)"]:.2f}% |
| Average P/L per Trade | Rs {average_pnl:,.2f} |
| Gross Profit | Rs {gross_profit:,.2f} |
| Gross Loss | Rs {gross_loss:,.2f} |

### Directional breakdown

| Direction | Trades | P/L |
|---|---:|---:|
| LONG | {long_trades} | Rs {long_pnl:,.2f} |
| SHORT | {short_trades} | Rs {short_pnl:,.2f} |
| TOTAL | {len(training_trades)} | Rs {total_pnl:,.2f} |

## 7. Mandatory Output

The program writes:

final_results.csv

with exactly these columns:

- Trade_ID
- Entry_Time
- Exit_Time
- Ticker
- Direction
- P&L
- Running_Balance

Trade IDs are sequential.

## 8. Hidden Evaluation Dataset

When hackathon_eval_secret.csv is placed in the same folder as main.py,
the program automatically detects it and runs final_strategy on it.

The resulting trades are written to final_results.csv.

If the hidden file is absent, the training dataset is used for local
pipeline testing.

## 9. Reproducibility and Look-Ahead Control

Each ticker is sorted chronologically before indicators are calculated.

A signal generated from day i uses information available through day i
only. The execution engine uses the next trading day's Open.

Shorts are opened at the next Open and closed at that same day's Close.

The hidden dataset is not used to fit, select, or tune the strategy.

## 10. Final Project Structure

The active trading strategy is final_strategy.

The earlier standalone strategies are no longer part of the strategy
registry. The project now produces one final portfolio and one mandatory
trade-results file.
"""


def run():
    # 1. Always run the training set so the report contains real metrics.
    print(f"Loading training data: {TRAIN_DATA}")

    train_df = load_dataset(TRAIN_DATA)
    strategy_func = strategies["final_strategy"]

    print("Running final_strategy on training data...")

    training_trades = clean_trade_output(
        run_backtest(train_df, strategy_func)
    )

    if len(training_trades) < 10:
        raise RuntimeError(
            "The rulebook requires at least 10 valid trades. "
            f"Training run generated {len(training_trades)}."
        )

    training_metrics = calculate_metrics(
        training_trades,
        "final_strategy"
    )

    REPORT_FILE.write_text(
        build_training_report(
            training_trades,
            training_metrics
        ),
        encoding="utf-8"
    )

    print(
        f"Training: {len(training_trades)} trades | "
        f"ROI {training_metrics['Net ROI (%)']:.2f}% | "
        f"Sharpe {training_metrics['Sharpe']:.2f} | "
        f"Max DD {training_metrics['Max DD (%)']:.2f}%"
    )

    # 2. The hidden file has priority for the mandatory final output.
    if EVAL_DATA.exists():
        eval_path = EVAL_DATA
    elif EVAL_DATA_FALLBACK.exists():
        eval_path = EVAL_DATA_FALLBACK
    else:
        eval_path = None

    if eval_path is not None:
        print(f"Hidden evaluation dataset found: {eval_path}")

        eval_df = load_dataset(eval_path)

        final_trades = clean_trade_output(
            run_backtest(eval_df, strategy_func)
        )

        if len(final_trades) < 10:
            raise RuntimeError(
                "The evaluation run generated fewer than 10 valid trades: "
                f"{len(final_trades)}."
            )

        print("Using hidden evaluation data for final_results.csv.")
    else:
        final_trades = training_trades
        print(
            "hackathon_eval_secret.csv not found. "
            "Using training data for local final_results.csv testing."
        )

    # 3. Mandatory competition output.
    final_trades.to_csv(FINAL_RESULTS, index=False)

    print(
        f"Generated {FINAL_RESULTS} with "
        f"{len(final_trades)} trades."
    )
    print(f"Generated training report: {REPORT_FILE}")


if __name__ == "__main__":
    run()
