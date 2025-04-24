import ccxt
import pandas as pd
import time

# Initialize exchange
exchange = ccxt.binanceus()

# Parameters
symbol = 'SOL/USDT'
timeframe = '1h'   # 1-hour candles
since = exchange.parse8601('2023-01-01T00:00:00Z')  # Adjust start date
limit = 1000       # Max candles per API call

# Fetch data in chunks
all_candles = []

while True:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not candles:
        break
    all_candles.extend(candles)
    since = candles[-1][0] + 1
    time.sleep(exchange.rateLimit / 1000)

# Convert to DataFrame
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save it
df.to_csv('solana_price_data.csv', index=False)
print(df.head())
