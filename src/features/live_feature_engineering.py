import os
import pandas as pd
from datetime import datetime

# =========================
# LOAD LIVE RAW DATA
# =========================

RAW_DATA_PATH = "data/raw/live_aqi_data.csv"

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"Live raw data file not found: {RAW_DATA_PATH}"
    )

df = pd.read_csv(RAW_DATA_PATH)

# =========================
# TIMESTAMP
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# =========================
# TIME FEATURES
# =========================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

# =========================
# AQI LAG FEATURES
# =========================

df["previous_aqi"] = df["aqi"].shift(1)

df["aqi_lag_3"] = df["aqi"].rolling(3).mean()
df["aqi_lag_6"] = df["aqi"].rolling(2).mean()
df["aqi_lag_12"] = df["aqi"].rolling(3).mean()

# =========================
# ROLLING FEATURES
# =========================

df["rolling_avg_3"] = df["aqi"].rolling(3).mean()
df["rolling_avg_6"] = df["aqi"].rolling(3).mean()

# =========================
# AQI CHANGE RATE
# =========================

df["aqi_change"] = df["aqi"].diff()

# =========================
# REMOVE NULLS
# =========================

df = df.dropna()

# =========================
# SAVE LIVE FEATURED DATA
# =========================

OUTPUT_PATH = "data/processed/live_featured_aqi_data.csv"

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Live feature engineering complete!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df)}")