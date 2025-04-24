import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the labeled data
df = pd.read_csv('/Users/stephenkussow/Desktop/senior research/solana_labeled.csv', parse_dates=['timestamp'])

# Use same features
features = ['rsi', 'ema_20', 'macd', 'macd_signal']
X = df[features]
y = df['target']

# Train/test split (no shuffling, to simulate time)
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_prices = df['close'][split_index:].reset_index(drop=True)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Simulate backtest
starting_cash = 1000
cash = starting_cash
holding = 0
position_open_price = 0
portfolio_values = []

for i in range(len(predictions)):
    price = test_prices[i]
    prediction = predictions[i]

    if prediction == 1 and cash > 0:
        # Buy
        holding = cash / price
        position_open_price = price
        cash = 0
    elif prediction == 0 and holding > 0:
        # Sell
        cash = holding * price
        holding = 0

    # Track portfolio value (even if holding)
    portfolio_value = cash if holding == 0 else holding * price
    portfolio_values.append(portfolio_value)

# Final value
final_value = portfolio_values[-1]
print(f"\n📈 Final portfolio value: ${final_value:.2f}")
print(f"💹 Return: {((final_value - starting_cash) / starting_cash) * 100:.2f}%")

# Compare to HODLing
buy_price = test_prices.iloc[0]
hodl_final = starting_cash * (test_prices.iloc[-1] / buy_price)
print(f"\n📊 HODL final value: ${hodl_final:.2f}")
print(f"HODL return: {((hodl_final - starting_cash) / starting_cash) * 100:.2f}%")

import matplotlib.pyplot as plt

# Create HODL performance line
hodl_values = [starting_cash * (price / test_prices.iloc[0]) for price in test_prices]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='AI Model Portfolio', linewidth=2)
plt.plot(hodl_values, label='Hold Strategy', linestyle='--')
plt.title('Backtest Performance: AI Trader vs Hold')
plt.xlabel('Time (candles)')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
