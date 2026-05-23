import os
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "models/best_aqi_model.pkl"

model = joblib.load(MODEL_PATH)

# =========================
# CREATE APP
# =========================

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="AQI Forecasting API using ML models",
    version="1.0"
)

# =========================
# INPUT SCHEMA
# =========================

class AQIInput(BaseModel):

    temperature: float
    humidity: float
    wind_speed: float
    pressure: float

    pm25: float
    pm10: float
    co: float
    no2: float
    o3: float

    hour: int
    day: int
    month: int
    day_of_week: int

    previous_aqi: float
    aqi_lag_3: float
    aqi_lag_6: float
    aqi_lag_12: float

    rolling_avg_3: float
    rolling_avg_6: float

    aqi_change: float

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def home():

    return {
        "message": "Pearls AQI Predictor API is running"
    }

# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict")

def predict(data: AQIInput):

    features = np.array([[
        data.temperature,
        data.humidity,
        data.wind_speed,
        data.pressure,

        data.pm25,
        data.pm10,
        data.co,
        data.no2,
        data.o3,

        data.hour,
        data.day,
        data.month,
        data.day_of_week,

        data.previous_aqi,
        data.aqi_lag_3,
        data.aqi_lag_6,
        data.aqi_lag_12,

        data.rolling_avg_3,
        data.rolling_avg_6,

        data.aqi_change
    ]])

    prediction = model.predict(features)[0]

    # AQI Category Logic

    if prediction <= 50:
        status = "Good"

    elif prediction <= 100:
        status = "Moderate"

    elif prediction <= 150:
        status = "Unhealthy for Sensitive Groups"

    elif prediction <= 200:
        status = "Unhealthy"

    elif prediction <= 300:
        status = "Very Unhealthy"

    else:
        status = "Hazardous"

    return {
        "predicted_aqi": round(float(prediction), 2),
        "status": status
    }