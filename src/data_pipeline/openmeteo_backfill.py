import pandas as pd
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

# =========================
# SETUP API CLIENT
# =========================

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

openmeteo = openmeteo_requests.Client(session=retry_session)

# =========================
# LOCATION
# =========================

LATITUDE = 24.8607
LONGITUDE = 67.0011

# Karachi coordinates

# =========================
# DATE RANGE
# =========================

END_DATE = datetime.today().date()
START_DATE = END_DATE - timedelta(days=90)

print(f"Fetching data from {START_DATE} to {END_DATE}")

# =========================
# WEATHER API
# =========================

weather_url = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": str(START_DATE),
    "end_date": str(END_DATE),
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "windspeed_10m",
        "surface_pressure"
    ],
    "timezone": "auto"
}

weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]

# =========================
# AIR QUALITY API
# =========================

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": str(START_DATE),
    "end_date": str(END_DATE),
    "hourly": [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone"
    ],
    "timezone": "auto"
}

air_response = openmeteo.weather_api(air_url, params=air_params)[0]

# =========================
# EXTRACT WEATHER DATA
# =========================

weather_hourly = weather_response.Hourly()

time_index = pd.date_range(
    start=pd.to_datetime(weather_hourly.Time(), unit="s"),
    end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s"),
    freq=pd.Timedelta(seconds=weather_hourly.Interval()),
    inclusive="left"
)

# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame({
    "timestamp": time_index,

    # Weather
    "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
    "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
    "wind_speed": weather_hourly.Variables(2).ValuesAsNumpy(),
    "pressure": weather_hourly.Variables(3).ValuesAsNumpy(),

    # Pollutants
    "pm25": air_response.Hourly().Variables(0).ValuesAsNumpy(),
    "pm10": air_response.Hourly().Variables(1).ValuesAsNumpy(),
    "co": air_response.Hourly().Variables(2).ValuesAsNumpy(),
    "no2": air_response.Hourly().Variables(3).ValuesAsNumpy(),
    "o3": air_response.Hourly().Variables(4).ValuesAsNumpy()
})

# =========================
# CREATE SIMPLE AQI
# =========================

df["aqi"] = (
    df["pm25"] * 0.5 +
    df["pm10"] * 0.2 +
    df["no2"] * 0.15 +
    df["o3"] * 0.15
)

# Round AQI
df["aqi"] = df["aqi"].round(2)

# =========================
# SAVE DATASET
# =========================

output_path = "data/raw/openmeteo_raw_data.csv"

df.to_csv(output_path, index=False)

print("Historical dataset created successfully!")
print(df.head())

print(f"Dataset shape: {df.shape}")