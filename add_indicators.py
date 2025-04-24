import pandas as pd
import ta  # Technical Analysis library

# Load CSV
df = pd.read_csv('/Users/stephenkussow/Desktop/senior research/solana_price_data.csv', parse_dates=['timestamp'])


# Sort by time (just in case)
df = df.sort_values('timestamp')

# Add RSI (Relative Strength Index)
df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()

# Add EMA (Exponential Moving Average)
df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()

# Add MACD (Moving Average Convergence Divergence)
macd = ta.trend.MACD(close=df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

# Drop NaN rows (from indicators needing lookback windows)
df = df.dropna()

# Save it
df.to_csv('/Users/stephenkussow/Desktop/senior research/solana_with_indicators.csv', index=False)
print(df.head())
