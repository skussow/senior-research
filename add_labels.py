import pandas as pd

# Load dataset with indicators
df = pd.read_csv('/Users/stephenkussow/Desktop/senior research/solana_with_indicators.csv', parse_dates=['timestamp'])

# Choose your prediction window (e.g., 3 hours ahead)
prediction_horizon = 3

# Add a target column: will the price go up in 'n' hours?
df['future_close'] = df['close'].shift(-prediction_horizon)
df['target'] = (df['future_close'] > df['close']).astype(int)

# Drop any rows with missing future values
df = df.dropna()

# Save it
df.to_csv('/Users/stephenkussow/Desktop/senior research/solana_labeled.csv', index=False)
print(df[['timestamp', 'close', 'future_close', 'target']].head(10))
