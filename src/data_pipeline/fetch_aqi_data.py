import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime


DATASET_PATH = "data/processed/aqi_dataset.csv"

# Load environment variables
load_dotenv()

# Get API key
API_KEY = os.getenv("AQICN_API_KEY")

# City to fetch AQI for
CITY = "Karachi"

# API URL
url = f"https://api.waqi.info/feed/{CITY}/?token={API_KEY}"

# Send request
response = requests.get(url)

# Convert response to JSON
data = response.json()

# Check API status
if data["status"] == "ok":

    # Extract useful information
    aqi_data = {
        "city": CITY,
        "aqi": data["data"]["aqi"],

        # Pollutants
        "pm25": data["data"]["iaqi"].get("pm25", {}).get("v"),
        "pm10": data["data"]["iaqi"].get("pm10", {}).get("v"),
        "no2": data["data"]["iaqi"].get("no2", {}).get("v"),
        "co": data["data"]["iaqi"].get("co", {}).get("v"),
        "o3": data["data"]["iaqi"].get("o3", {}).get("v"),

        # Weather
        "temperature": data["data"]["iaqi"].get("t", {}).get("v"),
        "humidity": data["data"]["iaqi"].get("h", {}).get("v"),
        "pressure": data["data"]["iaqi"].get("p", {}).get("v"),
        "wind_speed": data["data"]["iaqi"].get("w", {}).get("v"),

        # Metadata
        "timestamp": datetime.now()
    }

    # Convert to DataFrame
    df = pd.DataFrame([aqi_data])

    # Check if dataset already exists
    if os.path.exists(DATASET_PATH):

        # Load existing dataset
        existing_df = pd.read_csv(DATASET_PATH)

        # Append new row
        updated_df = pd.concat([existing_df, df], ignore_index=True)

    else:
        # Create new dataset
        updated_df = df

    # Save updated dataset
    updated_df.to_csv(DATASET_PATH, index=False)

    print("AQI data fetched successfully!")
    print(df)

else:
    print("Failed to fetch AQI data")
    print(data)