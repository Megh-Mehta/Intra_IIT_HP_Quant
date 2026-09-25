import talib
import pandas as pd
import numpy as np

def strat_long_ema_crossover(group: pd.DataFrame) -> pd.DataFrame:
    """Long Strategy: Hyper-Fast 3/8 EMA Crossover"""
    group = group.copy()
    
    # 1. Calculate Faster Indicators
    group['EMA_3'] = talib.EMA(group['aC'], timeperiod=3)
    group['EMA_8'] = talib.EMA(group['aC'], timeperiod=8)
    
    # 2. Generate Signals based on 3 and 8
    group['Buy_Signal'] = np.where(
        (group['EMA_3'] > group['EMA_8']) & 
        (group['EMA_3'].shift(1) <= group['EMA_8'].shift(1)), 1, 0
    )
        
    group['Exit_Signal'] = np.where(
        (group['EMA_3'] < group['EMA_8']) & 
        (group['EMA_3'].shift(1) >= group['EMA_8'].shift(1)), 1, 0
    )
    
    # 3. Conviction Score (Spread between 3 and 8)
    spread = ((group['EMA_3'] - group['EMA_8']) / group['EMA_8']) * 100
    group['Conviction_Score'] = np.where(group['Buy_Signal'] == 1, spread, 0)
        
    return group

def strat_long_bollinger(group: pd.DataFrame) -> pd.DataFrame:
    """Long Strategy: Bollinger Bands + RSI Oversold Mean Reversion"""
    group = group.copy()
    
    # 1. Calculate Indicators
    # BBANDS generates the upper, middle (average), and lower statistical bands
    group['upper'], group['middle'], group['lower'] = talib.BBANDS(
        group['aC'], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # RSI measures momentum (0-100) to confirm if a stock is genuinely oversold
    group['RSI'] = talib.RSI(group['aC'], timeperiod=10)
    
    # 2. Generate Binary Signals
    # BUY: The stock drops below its lower statistical band AND momentum is dead (RSI < 40)
    group['Buy_Signal'] = np.where(
        (group['aC'] < group['lower']) & (group['RSI'] < 40), 1, 0
    )
        
    # EXIT: The stock snaps back to its middle average price
    group['Exit_Signal'] = np.where(
        (group['aC'] > group['middle']), 1, 0
    )
    
    # 3. Calculate Conviction Score (Bypasses Alphabetical Sorting)
    # Measures how far (in percentage) the price dropped below the lower band.
    crash_depth = ((group['lower'] - group['aC']) / group['lower']) * 100
    
    # Only assign the score on days where a Buy_Signal is actually triggered
    group['Conviction_Score'] = np.where(group['Buy_Signal'] == 1, crash_depth, 0)
        
    return group

strategies = {
    "EMA_Crossover": strat_long_ema_crossover,
    "Bollinger_Bands": strat_long_bollinger,
}