import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data():
    """Generate realistic stock data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')[:100]
    np.random.seed(42)
    
    # Generate realistic price data with trend
    prices = []
    price = 100
    for i in range(100):
        change = np.random.normal(0, 2)
        price = price + change
        prices.append(max(price, 10))  # Ensure price doesn't go below 10
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': prices[i] + np.random.normal(0, 0.5),
            'high': prices[i] + abs(np.random.normal(1, 0.5)),
            'low': prices[i] - abs(np.random.normal(1, 0.5)),
            'close': prices[i],
            'volume': np.random.randint(100000, 1000000)
        })
    
    return data

def calculate_technical_indicators(data):
    """Calculate RSI, MACD, Moving Averages"""
    df = pd.DataFrame(data)
    df['close'] = pd.to_numeric(df['close'])
    
    # Simple Moving Averages
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df.to_dict('records')
