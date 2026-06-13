import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/yield_df.csv')

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 0'])

# Encode categorical columns
df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

# Features and target
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.3)
plt.xlabel("Actual Yield (hg/ha)")
plt.ylabel("Predicted Yield (hg/ha)")
plt.title("Actual vs Predicted Crop Yield")
plt.savefig("actual_vs_predicted.png")
print("Saved actual_vs_predicted.png")

# Feature importance (top 15)
plt.figure(figsize=(8,6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature_importance.png")



import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')
print("Model saved.")