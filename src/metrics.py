import pandas as pd
import numpy as np

def calculate_metrics(results_df: pd.DataFrame, strat_name: str, initial_capital: float = 1000000.0) -> dict:
    # Handle cases where the strategy generated zero trades
    if results_df.empty:
        return {
            "Strategy": strat_name, "Trades": 0, "Net ROI (%)": 0.0, 
            "Sharpe": 0.0, "Max DD (%)": 0.0, "Profit Factor": 0.0, "Win Rate (%)": 0.0
        }

    total_trades = len(results_df)
    final_balance = results_df['Running_Balance'].iloc[-1]
    
    # 1. Net ROI
    net_roi = ((final_balance - initial_capital) / initial_capital) * 100

    # 2. Maximum Drawdown (calculates the deepest percentage drop from an all-time high)
    peaks = pd.concat([pd.Series([initial_capital]), results_df['Running_Balance']]).cummax()
    drawdowns = (peaks - pd.concat([pd.Series([initial_capital]), results_df['Running_Balance']])) / peaks
    max_drawdown = drawdowns.max() * 100

    # 3. Profit Factor (Total Gross Profit / Total Gross Loss)
    gross_profit = results_df[results_df['P&L'] > 0]['P&L'].sum()
    gross_loss = abs(results_df[results_df['P&L'] < 0]['P&L'].sum())
    profit_factor = float('inf') if gross_loss == 0 else (gross_profit / gross_loss)

    # 4. Win Rate
    wins = len(results_df[results_df['P&L'] > 0])
    win_rate = (wins / total_trades) * 100

    # 5. Sharpe Ratio (Risk-adjusted return, annualized over 252 trading days)
    results_df_copy = results_df.copy()
    results_df_copy['Exit_Date'] = pd.to_datetime(results_df_copy['Exit_Time']).dt.date
    daily_pnl = results_df_copy.groupby('Exit_Date')['P&L'].sum()
    daily_balance = initial_capital + daily_pnl.cumsum()
    daily_returns = daily_pnl / pd.Series([initial_capital] + daily_balance.tolist()[:-1], index=daily_balance.index)
    sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0.0

    return {
        "Strategy": strat_name,
        "Trades": total_trades,
        "Net ROI (%)": round(net_roi, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD (%)": round(max_drawdown, 2),
        "Profit Factor": round(profit_factor, 2),
        "Win Rate (%)": round(win_rate, 2)
    }