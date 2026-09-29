# Final Strategy Report

This file documents the final trading system for the Intra IIT Tech Meet 2026
Day-Trading Quant Hackathon.

The authoritative training metrics in this report are generated automatically
by `main.py` from the current training dataset. Running:

```bash
python main.py
```

replaces this file with the exact measured training P/L, ROI, Sharpe ratio,
maximum drawdown, profit factor, win rate, trade count, and long/short
breakdown.

## Final strategy

The active strategy is **final_strategy**.

It combines:

1. Statistical Mean Reversion
2. Causal Bollinger + Wilder RSI

They are treated as **one portfolio**, not two separate portfolios.

### Statistical Mean Reversion

- 20-day rolling mean
- 20-day rolling standard deviation
- Z-score
- RSI(10)

Long entry:
- Z-score < -2
- RSI < 35

Long exit:
- Z-score > -0.25

Short entry:
- Z-score > +2
- RSI > 65

Shorts are closed on the same trading day.

### Causal Bollinger + Wilder RSI

- 15-day Bollinger Band lookback
- 2 standard deviations
- 14-period Wilder RSI

Long entry:
- Price below lower Bollinger Band
- RSI < 40

Long exit:
- Price reaches the middle Bollinger Band

Short entry:
- Price above upper Bollinger Band
- RSI > 60

Shorts are closed on the same trading day.

## Signal combination

The components are combined per ticker and date.

- If at least one component signals LONG and neither signals SHORT, the
  final strategy can enter LONG.
- If at least one component signals SHORT and neither signals LONG, the
  final strategy can enter SHORT.
- If the components disagree, no new position is opened.
- Conviction is the maximum conviction from the two components.

This produces one combined trade stream and one portfolio.

## Why the strategy is intended to work

Both components attempt to identify unusually extended prices relative to
recent history.

The statistical component uses a z-score to quantify the size of the
deviation from a rolling mean.

The Bollinger/RSI component provides a second measurement of the same broad
phenomenon using volatility bands and momentum/extremeness.

The combination can therefore require agreement in market direction before
entering when the two components disagree, while still allowing either
component to identify an extreme independently when the other component is
neutral.

This is a testable statistical hypothesis, not a guarantee of profitability
on unseen data.

## Rulebook compliance

The implementation follows the supplied rulebook:

- Initial balance: Rs 10,00,000.
- Entry brokerage: 0.05%.
- Exit brokerage: 0.05%.
- Individual trade value is capped at 25% of current portfolio value.
- Quantity is an integer.
- Quantity is capped at 10% of the signal day's reported volume.
- Long positions can be carried.
- Short positions are closed at the same day's Close.
- A signal from day i executes at the next trading day's Open.
- Signals use only current and historical information.
- No external API calls are used during execution.
- The hidden evaluation dataset is not used to tune the strategy.

## Required output

The program generates:

`final_results.csv`

with exactly:

- Trade_ID
- Entry_Time
- Exit_Time
- Ticker
- Direction
- P&L
- Running_Balance

The hidden evaluation file is expected at:

`hackathon_eval_secret.csv`

in the same directory as `main.py`.

When that file exists, it takes priority for generating
`final_results.csv`.

If it does not exist, the training dataset is used so the complete pipeline
can be tested locally.

## Training metrics

Run `python main.py` to populate this section with the actual training
results from the current code and dataset.

The generated report includes:

- Total trades
- Net P/L
- Net ROI
- Sharpe ratio
- Maximum drawdown
- Profit factor
- Win rate
- Average P/L per trade
- Gross profit
- Gross loss
- LONG trade count and P/L
- SHORT trade count and P/L

The metrics are deliberately generated at runtime rather than hard-coded so
that the report remains synchronized with the actual implementation.

## Project structure

```text
Intra_IIT_HP_Quant/
├── data/
│   └── hackathon_train_data.csv
├── src/
│   ├── engine.py
│   ├── metrics.py
│   └── signals.py
├── main.py
├── requirements.txt
├── FINAL_STRATEGY_REPORT.md
└── final_results.csv        # generated at runtime
```

The earlier standalone strategies are no longer registered. The strategy
registry now contains only `final_strategy`.
