import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load featured dataset
df = pd.read_csv("data/processed/featured_aqi_dataset.csv")

# Select features
features = [
    "aqi",
    "pm25",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "hour",
    "day",
    "month",
    "day_of_week",
    "previous_aqi",
    "rolling_avg_3",
    "aqi_change"
]

# Input features
X = df[features]

# Target
y = df["future_aqi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save model
joblib.dump(model, "models/random_forest_aqi_model.pkl")

# Print results
print("Model Training Completed!")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")