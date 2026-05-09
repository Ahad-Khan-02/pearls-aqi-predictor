import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/aqi_dataset.csv")

# Convert timestamp column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by timestamp
df = df.sort_values("timestamp")

# =========================
# TIME FEATURES
# =========================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

# =========================
# LAG FEATURES
# =========================

df["previous_aqi"] = df["aqi"].shift(1)  #future AQI depends on past AQI

# Rolling average
df["rolling_avg_3"] = df["aqi"].rolling(window=3, min_periods=1).mean()

# AQI change
df["aqi_change"] = df["aqi"].diff()

# Fill missing pollutants
pollutant_columns = ["pm10", "no2", "co", "o3"]

df[pollutant_columns] = df[pollutant_columns].fillna(0)

# =========================
# TARGET VARIABLE
# =========================

df["future_aqi"] = df["aqi"].shift(-1)

# Remove essential nulls only
df = df.dropna(subset=["future_aqi", "previous_aqi"])

# Save engineered dataset
df.to_csv("data/processed/featured_aqi_dataset.csv", index=False)

print("Feature engineering completed!")
print(df.head())