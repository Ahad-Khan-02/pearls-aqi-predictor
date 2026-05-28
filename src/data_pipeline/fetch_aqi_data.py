import os
import pandas as pd
from datetime import datetime
import requests

# =========================
# CONFIG
# =========================

CITY = "karachi"
API_URL = f"https://api.waqi.info/feed/{CITY}/?token=demo"

OUTPUT_PATH = "data/raw/live_aqi_data.csv"

# =========================
# FETCH LIVE DATA
# =========================

response = requests.get(API_URL)

if response.status_code != 200:
    raise Exception(
        f"API request failed: {response.status_code}"
    )

data = response.json()

if data["status"] != "ok":
    raise Exception(
        f"AQI API error: {data}"
    )

aqi_data = data["data"]

# =========================
# EXTRACT FEATURES
# =========================

record = {
    "timestamp": datetime.now(),

    "aqi": aqi_data.get("aqi"),

    "temperature":
        aqi_data.get("iaqi", {})
        .get("t", {})
        .get("v"),

    "humidity":
        aqi_data.get("iaqi", {})
        .get("h", {})
        .get("v"),

    "pressure":
        aqi_data.get("iaqi", {})
        .get("p", {})
        .get("v"),

    "wind_speed":
        aqi_data.get("iaqi", {})
        .get("w", {})
        .get("v"),

    "pm25":
        aqi_data.get("iaqi", {})
        .get("pm25", {})
        .get("v"),

    "pm10":
        aqi_data.get("iaqi", {})
        .get("pm10", {})
        .get("v"),

    "co":
        aqi_data.get("iaqi", {})
        .get("co", {})
        .get("v"),

    "no2":
        aqi_data.get("iaqi", {})
        .get("no2", {})
        .get("v"),

    "o3":
        aqi_data.get("iaqi", {})
        .get("o3", {})
        .get("v"),
}

# =========================
# CREATE DATAFRAME
# =========================

new_df = pd.DataFrame([record])

# =========================
# APPEND TO CSV
# =========================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

if os.path.exists(OUTPUT_PATH):

    existing_df = pd.read_csv(OUTPUT_PATH)

    updated_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

else:
    updated_df = new_df

# =========================
# SAVE
# =========================

updated_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Live AQI data fetched successfully!")
print(updated_df.tail())