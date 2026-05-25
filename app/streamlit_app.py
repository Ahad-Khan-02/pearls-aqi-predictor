import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import sys
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from src.utils.fetch_latest_features import get_latest_features

# =========================
# LOAD TRAINED MODEL
# =========================

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "best_aqi_model.pkl"
)

model = joblib.load(model_path)

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

sample_data = get_latest_features()
# Remove timestamp immediately
sample_data.pop("timestamp", None)

# =========================
# CALL FASTAPI
# =========================
payload = sample_data.copy()
# Remove timestamp
payload.pop("timestamp", None)

try:
    response = requests.post(
        API_URL,
        json=payload
    )

    prediction = response.json()

    predicted_aqi = prediction["predicted_aqi"]
    status = prediction["status"]

except Exception as e:
    st.error(f"API connection failed: {e}")

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

    try:

        hours = np.arange(72)

        forecast_values = []

        current_input = payload.copy()

        for i in range(72):

            # Remove timestamp if exists
            current_input.pop("timestamp", None)

            response = requests.post(
                API_URL,
                json=current_input
            )

            prediction_json = response.json()

            next_aqi = prediction_json["predicted_aqi"]

            forecast_values.append(next_aqi)

            # =========================
            # UPDATE FEATURES
            # =========================

            previous_value = current_input["previous_aqi"]

            current_input["aqi_change"] = (
                next_aqi - previous_value
            )

            current_input["previous_aqi"] = next_aqi

            current_input["aqi_lag_3"] = next_aqi
            current_input["aqi_lag_6"] = next_aqi
            current_input["aqi_lag_12"] = next_aqi

            current_input["rolling_avg_3"] = (
                current_input["rolling_avg_3"] + next_aqi
            ) / 2

            current_input["rolling_avg_6"] = (
                current_input["rolling_avg_6"] + next_aqi
            ) / 2

            current_input["hour"] = (
                current_input["hour"] + 1
            ) % 24

        forecast_df = pd.DataFrame({
            "Hour": hours,
            "AQI": forecast_values
        })

        fig = px.line(
            forecast_df,
            x="Hour",
            y="AQI",
            title="72-Hour AQI Forecast"
        )

        st.plotly_chart(fig, width="stretch")

    except Exception as e:

        st.error(f"Forecast generation failed: {e}")

# =========================
# EXPLAINABILITY PAGE
# =========================

elif page == "Explainability":

    st.subheader("SHAP Feature Importance")

    # =========================
    # CREATE DATAFRAME
    # =========================

    input_df = pd.DataFrame([payload])

    # =========================
    # SHAP EXPLAINER
    # =========================

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": shap_importance
    })

    shap_df = shap_df.sort_values(
        by="Importance",
        ascending=True
    )

    # =========================
    # PLOT
    # =========================

    fig = px.bar(
        shap_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Real SHAP Feature Importance"
    )

    st.plotly_chart(fig, width="stretch")

    # =========================
    # SHOW TOP FACTORS
    # =========================

    st.markdown("### Top Influencing Features")

    top_features = shap_df.sort_values(
        by="Importance",
        ascending=False
    ).head(5)

    st.table(top_features)

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