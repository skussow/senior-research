import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load labeled data
df = pd.read_csv('/Users/stephenkussow/Desktop/senior research/solana_labeled.csv', parse_dates=['timestamp'])

# Define features and target
features = ['rsi', 'ema_20', 'macd', 'macd_signal']
X = df[features]
y = df['target']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Plot feature importances
importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
