import os
import pandas as pd
from datetime import datetime

 
# load live raw data
RAW_DATA_PATH = "data/raw/live_aqi_data.csv"

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"Live raw data file not found: {RAW_DATA_PATH}"
    )

df = pd.read_csv(RAW_DATA_PATH)

 
# timezone conversion
df["timestamp"] = pd.to_datetime(df["timestamp"])

# time features
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

 
# aqi lag features
df["previous_aqi"] = df["aqi"].shift(1)

df["aqi_lag_3"] = df["aqi"].rolling(3).mean()
df["aqi_lag_6"] = df["aqi"].rolling(2).mean()
df["aqi_lag_12"] = df["aqi"].rolling(3).mean()


# rolling averages
df["rolling_avg_3"] = df["aqi"].rolling(3).mean()
df["rolling_avg_6"] = df["aqi"].rolling(3).mean()
df["rolling_avg_24"] = df["aqi"].rolling(window=24, min_periods=1).mean()

 
# aqi change and trend
df["aqi_change"] = df["aqi"].diff()
df["aqi_trend"] = df["previous_aqi"] - df["aqi_lag_3"]

 
# pollution index (simple weighted sum of pollutants)
 
df["pollution_index"] = (
    df["pm25"] * 0.5 +
    df["pm10"] * 0.3 +
    df["no2"] * 0.2
)

 
# remove nulls
df = df.dropna()
 
# save featured dataset to csv
OUTPUT_PATH = "data/processed/live_featured_aqi_data.csv"

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Live feature engineering complete!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df)}")