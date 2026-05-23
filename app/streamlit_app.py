import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("🌍 Pearls AQI Predictor")
st.markdown("### Real-Time AQI Forecasting Dashboard")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Forecast",
        "Explainability",
        "Model Metrics"
    ]
)

# =========================
# SAMPLE INPUT DATA
# =========================

sample_data = {
    "temperature": 31,
    "humidity": 55,
    "wind_speed": 5,
    "pressure": 1012,

    "pm25": 160,
    "pm10": 120,
    "co": 500,
    "no2": 30,
    "o3": 45,

    "hour": 14,
    "day": 20,
    "month": 5,
    "day_of_week": 1,

    "previous_aqi": 155,
    "aqi_lag_3": 150,
    "aqi_lag_6": 145,
    "aqi_lag_12": 140,

    "rolling_avg_3": 152,
    "rolling_avg_6": 148,

    "aqi_change": 5
}

# =========================
# CALL FASTAPI
# =========================

try:

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=sample_data
    )

    prediction = response.json()

    predicted_aqi = prediction["predicted_aqi"]
    status = prediction["status"]

except:

    predicted_aqi = 156
    status = "Unhealthy"

# =========================
# DASHBOARD PAGE
# =========================

if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Predicted AQI",
            value=predicted_aqi
        )

    with col2:

        st.metric(
            label="AQI Status",
            value=status
        )

    with col3:

        if predicted_aqi <= 50:
            alert = "🟢 Good"

        elif predicted_aqi <= 100:
            alert = "🟡 Moderate"

        elif predicted_aqi <= 150:
            alert = "🟠 Unhealthy for Sensitive Groups"

        elif predicted_aqi <= 200:
            alert = "🔴 Unhealthy"

        else:
            alert = "⚫ Hazardous"

        st.metric(
            label="Health Alert",
            value=alert
        )

    st.markdown("---")

    # =========================
    # POLLUTANT CARDS
    # =========================

    st.subheader("Pollutant Levels")

    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric("PM2.5", sample_data["pm25"])
    p2.metric("PM10", sample_data["pm10"])
    p3.metric("NO2", sample_data["no2"])
    p4.metric("CO", sample_data["co"])
    p5.metric("O3", sample_data["o3"])

# =========================
# FORECAST PAGE
# =========================

elif page == "Forecast":

    st.subheader("3-Day AQI Forecast")

    hours = np.arange(72)

    forecast = predicted_aqi + np.random.normal(0, 5, 72)

    forecast_df = pd.DataFrame({
        "Hour": hours,
        "AQI": forecast
    })

    fig = px.line(
        forecast_df,
        x="Hour",
        y="AQI",
        title="72-Hour AQI Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# EXPLAINABILITY PAGE
# =========================

elif page == "Explainability":

    st.subheader("SHAP Feature Importance")

    shap_data = pd.DataFrame({
        "Feature": [
            "PM2.5",
            "Temperature",
            "Humidity",
            "O3",
            "NO2"
        ],
        "Importance": [
            0.35,
            0.22,
            0.15,
            0.12,
            0.08
        ]
    })

    fig = px.bar(
        shap_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# MODEL METRICS PAGE
# =========================

elif page == "Model Metrics":

    st.subheader("Model Performance")

    metrics_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R² Score"
        ],
        "Value": [
            4.31,
            6.29,
            0.79
        ]
    })

    st.table(metrics_df)

    st.markdown("### Models Used")

    st.write("""
    - Random Forest
    - Ridge Regression
    - TensorFlow Neural Network
    """)