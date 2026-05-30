import sys
import os
import tempfile

# Windows-safe temp directory for Hopsworks
tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

# Create tmp directory if missing
os.makedirs(tempfile.tempdir, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import streamlit as st
from src.utils.fetch_latest_features import get_latest_features
import joblib
import pandas as pd
import numpy as np


# PAGE CONFIG

st.set_page_config(
    page_title="Karachi AQI Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD SHARED CSS

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# LOAD MODEL + DATA (cached)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_aqi_model.pkl")
    return joblib.load(model_path)

@st.cache_data(ttl=3600)
def load_features():
    return get_latest_features()

FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week","is_weekend", "is_rush_hour",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6","rolling_avg_24", "aqi_change","aqi_trend", "pollution_index"
]

model     = load_model()
raw_data  = load_features()
payload   = {k: raw_data[k] for k in FEATURE_COLUMNS}

# Prediction
input_df      = pd.DataFrame([payload])
predicted_aqi = round(float(model.predict(input_df)[0]), 2)

# AQI category
def get_aqi_category(aqi):
    if aqi <= 50:   return "Good",                        "#00e676", "🟢"
    if aqi <= 100:  return "Moderate",                    "#ffeb3b", "🟡"
    if aqi <= 150:  return "Unhealthy for Sensitive",     "#ff9800", "🟠"
    if aqi <= 200:  return "Unhealthy",                   "#f44336", "🔴"
    if aqi <= 300:  return "Very Unhealthy",              "#9c27b0", "🟣"
    return "Hazardous",                                   "#b71c1c", "⚫"

status, color, icon = get_aqi_category(predicted_aqi)


# AQI ALERT SYSTEM

if predicted_aqi > 300:

    st.error(
        "HAZARDOUS AQI ALERT — Avoid all outdoor activity. Health emergency conditions detected."
    )

elif predicted_aqi > 200:

    st.error(
        "VERY UNHEALTHY AIR — Serious health effects possible. Stay indoors if possible."
    )

elif predicted_aqi > 150:

    st.warning(
        "UNHEALTHY AQI — Sensitive groups should avoid prolonged outdoor exposure."
    )

elif predicted_aqi > 100:

    st.warning(
        "Moderate pollution detected — Consider wearing a mask outdoors."
    )

else:

    st.success(
        "✅ Air quality is currently acceptable."
    )

# Store in session for pages to use
st.session_state["model"]         = model
st.session_state["raw_data"]      = raw_data
st.session_state["payload"]       = payload
st.session_state["predicted_aqi"] = predicted_aqi
st.session_state["status"]        = status
st.session_state["color"]         = color
st.session_state["icon"]          = icon
st.session_state["FEATURE_COLUMNS"] = FEATURE_COLUMNS

 
# HERO SECTION
 

st.markdown(f"""
<div class="hero-section">
    <div class="hero-badge" style="color:{color}; border-color:{color};">
        {icon} LIVE
    </div>
    <h1 class="hero-title">Karachi AQI Intelligence</h1>
    <p class="hero-subtitle">Real-time air quality monitoring & 72-hour forecasting</p>
    <div class="hero-aqi" style="color:{color};">
        {predicted_aqi}
        <span class="hero-aqi-label">AQI</span>
    </div>
    <div class="hero-status" style="background:{color}22; border:1px solid {color}44; color:{color};">
        {status}
    </div>
</div>
""", unsafe_allow_html=True)


# TOP METRICS ROW

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("🌡️ Temperature", f"{raw_data.get('temperature', 'N/A')}°C"),
    ("💧 Humidity",    f"{raw_data.get('humidity', 'N/A')}%"),
    ("💨 Wind Speed",  f"{raw_data.get('wind_speed', 'N/A')} km/h"),
    ("🔵 PM2.5",       f"{raw_data.get('pm25', 'N/A')} µg/m³"),
    ("🟤 PM10",        f"{raw_data.get('pm10', 'N/A')} µg/m³"),
]

for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

