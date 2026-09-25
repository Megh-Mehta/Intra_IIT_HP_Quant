import os
import pandas as pd
from src.signals import strategies
from src.engine import run_backtest
from src.metrics import calculate_metrics

def main():
    # 1. Load the raw dataset
    data_path = 'data/hackathon_train_data.csv'
    print(f"Loading market data from: {data_path}")
    df_raw = pd.read_csv(data_path)
    df_raw['dt'] = pd.to_datetime(df_raw['dt'])

    # 2. Ensure the reports directory exists so the script doesn't crash on export
    os.makedirs('reports', exist_ok=True)

    # 3. Loop through every strategy defined in the signals.py dictionary
    print("Running Backtests...")
    comparison_results = []
    
    for name, func in strategies.items():
        print(f" -> Testing {name}...")
        
        # Run execution engine for this specific strategy
        results_df = run_backtest(df_raw, func)
        
        # Automatically export the individual trade log
        if not results_df.empty:
            output_path = f'reports/{name}.csv'
            results_df.to_csv(output_path, index=False)
            
        # Calculate scorecard metrics and save them for the master table
        metrics = calculate_metrics(results_df, name)
        comparison_results.append(metrics)

    # 4. Display the Final Comparison Scoreboard
    print("\n" + "="*85)
    print("STRATEGY COMPARISON SCOREBOARD")
    print("="*85)
    comp_df = pd.DataFrame(comparison_results).set_index("Strategy")
    print(comp_df.to_string())
    print("="*85)

if __name__ == '__main__':
    main()