import pandas as pd

def run_backtest(df_raw: pd.DataFrame, strategy_func) -> pd.DataFrame:
    processed_dfs = []
    for _, group_df in df_raw.groupby('tkr'):
        processed_dfs.append(strategy_func(group_df))
    df = pd.concat(processed_dfs, ignore_index=True)
    df = df.sort_values(['dt', 'tkr']).reset_index(drop=True)

    initial_capital = 1000000.0
    balance = initial_capital
    cumulative_pnl = 0.0

    brokerage_rate = 0.0005 
    max_portfolio_pct = 0.15 
    max_volume_pct = 0.10 
    trailing_stop_pct = 0.05 

    positions = {} 
    trade_history = []
    trade_id_counter = 1
    unique_dates = df['dt'].sort_values().unique()

    # 3. T+1 Execution Loop
    for i in range(len(unique_dates) - 1):
        today_date = unique_dates[i]
        tomorrow_date = unique_dates[i + 1] 
        today_data = df[df['dt'] == today_date]
        tomorrow_data = df[df['dt'] == tomorrow_date]
        
        # --- A. Process Exits ---
        tickers_to_remove = []
        for tkr, pos in positions.items():
            today_ticker_data = today_data[today_data['tkr'] == tkr]
            if today_ticker_data.empty: continue
            row = today_ticker_data.iloc[0]
            
            # Update trailing stop high
            if row['aH'] > pos['highest_aH']:
                positions[tkr]['highest_aH'] = row['aH']
            stop_price = positions[tkr]['highest_aH'] * (1 - trailing_stop_pct)
            
            # Use .get() to default to 0 if signal columns are missing
            if row['aC'] < stop_price or row.get('Exit_Signal', 0) == 1:
                tomorrow_ticker_data = tomorrow_data[tomorrow_data['tkr'] == tkr]
                if tomorrow_ticker_data.empty: continue
                exec_row = tomorrow_ticker_data.iloc[0]
                
                exit_price = exec_row['aO'] 
                qty = pos['qty']
                
                net_exit = (exit_price * qty) * (1 - brokerage_rate)
                entry_cost = (pos['entry_price'] * qty) * (1 + brokerage_rate)
                pnl = net_exit - entry_cost
                
                balance += net_exit
                cumulative_pnl += pnl
                realized_equity = initial_capital + cumulative_pnl
                
                trade_history.append({
                    'Trade_ID': trade_id_counter,
                    'Entry_Time': pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit_Time': exec_row['dt'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker': tkr,
                    'Direction': 'LONG',
                    'P&L': round(pnl, 2),
                    'Running_Balance': round(realized_equity, 2)
                })
                trade_id_counter += 1
                tickers_to_remove.append(tkr)
                
        for tkr in tickers_to_remove:
            del positions[tkr]

        if 'Conviction_Score' not in today_data.columns:
            today_data['Conviction_Score'] = 0
            
        # Sort the data so the highest conviction scores are processed first
        today_data = today_data.sort_values(by='Conviction_Score', ascending=False)
            
        # --- B. Process Entries ---
        for _, row in today_data.iterrows():
            tkr = row['tkr']
            if tkr not in positions and row.get('Buy_Signal', 0) == 1:
                tomorrow_ticker_data = tomorrow_data[tomorrow_data['tkr'] == tkr]
                if tomorrow_ticker_data.empty: continue
                exec_row = tomorrow_ticker_data.iloc[0]
                
                entry_price = exec_row['aO'] 
                max_invest = balance * max_portfolio_pct
                qty_balance = int(max_invest / (entry_price * (1 + brokerage_rate)))
                qty_vol = int(exec_row['aV'] * max_volume_pct)
                qty = min(qty_balance, qty_vol)
                
                if qty > 0:
                    total_deduction = (qty * entry_price) * (1 + brokerage_rate)
                    if balance >= total_deduction:
                        balance -= total_deduction
                        positions[tkr] = {
                            'qty': qty,
                            'entry_price': entry_price,
                            'highest_aH': entry_price,
                            'entry_time': exec_row['dt']
                        }

    # 4. Force Close at End of Dataset
    last_date = unique_dates[-1]
    last_data = df[df['dt'] == last_date]
    for tkr, pos in positions.items():
        last_ticker_data = last_data[last_data['tkr'] == tkr]
        if not last_ticker_data.empty:
            exec_row = last_ticker_data.iloc[0]
            qty = pos['qty']
            
            net_exit = (exec_row['aC'] * qty) * (1 - brokerage_rate)
            entry_cost = (pos['entry_price'] * qty) * (1 + brokerage_rate)
            pnl = net_exit - entry_cost
            
            balance += net_exit
            cumulative_pnl += pnl
            realized_equity = initial_capital + cumulative_pnl
            
            trade_history.append({
                'Trade_ID': trade_id_counter,
                'Entry_Time': pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Exit_Time': exec_row['dt'].strftime('%Y-%m-%d %H:%M:%S'),
                'Ticker': tkr,
                'Direction': 'LONG',
                'P&L': round(pnl, 2),
                'Running_Balance': round(realized_equity, 2)
            })
            trade_id_counter += 1

    return pd.DataFrame(trade_history)