import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/random_forest_aqi_model.pkl")

# Load latest featured dataset
df = pd.read_csv("data/processed/featured_aqi_dataset.csv")

# Feature columns
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

# Get latest row
latest_data = df[features].iloc[-1:]

# Make prediction
prediction = model.predict(latest_data)

print("AQI Forecast for Next 3 Days:")
print(f"Predicted AQI: {prediction[0]:.2f}")