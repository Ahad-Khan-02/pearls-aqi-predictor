import pandas as pd

# =========================
# LOAD RAW DATA
# =========================

df = pd.read_csv("data/raw/openmeteo_raw_data.csv")

# =========================
# DATETIME PROCESSING
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp")

# =========================
# TIME FEATURES
# =========================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(
    lambda x: 1 if x >= 5 else 0
)
df["is_rush_hour"] = df["hour"].apply(
    lambda x: 1 if 7 <= x <= 10 or 17 <= x <= 20 else 0
)

# =========================
# LAG FEATURES
# =========================

df["previous_aqi"] = df["aqi"].shift(1)
df["aqi_lag_3"] = df["aqi"].shift(3)
df["aqi_lag_6"] = df["aqi"].shift(6)
df["aqi_lag_12"] = df["aqi"].shift(12)

# =========================
# ROLLING FEATURES
# =========================

df["rolling_avg_3"] = df["aqi"].rolling(window=3, min_periods=1).mean()
df["rolling_avg_6"] = df["aqi"].rolling(window=6, min_periods=1).mean()
df["rolling_avg_24"] = df["aqi"].rolling(window=24, min_periods=1).mean()

# =========================
# AQI CHANGE RATE
# =========================

df["aqi_change"] = df["aqi"].diff()
df["aqi_trend"] = df["previous_aqi"] - df["aqi_lag_3"]

# =========================
# POLLUTION INTENSITY
# =========================
df["pollution_index"] = (
    df["pm25"] * 0.5 +
    df["pm10"] * 0.3 +
    df["no2"] * 0.2
)


# =========================
# TARGET VARIABLE
# =========================

# Predict AQI 72 hours later
df["future_aqi"] = df["aqi"].shift(-72)

# =========================
# REMOVE NULLS
# =========================

df = df.dropna()

# =========================
# SAVE FEATURED DATASET
# =========================

output_path = "data/processed/featured_aqi_data.csv"

df.to_csv(output_path, index=False)

print("Feature engineering completed successfully!")
print(df.head())

print(f"Featured dataset shape: {df.shape}")