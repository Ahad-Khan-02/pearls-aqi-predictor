# Pearls AQI Predictor

An end-to-end Air Quality Index (AQI) forecasting system built using Machine Learning, Hopsworks, GitHub Actions, and Streamlit.

The system automatically collects environmental data, generates features, trains forecasting models, stores models in a Model Registry, and provides real-time AQI monitoring and 72-hour forecasting through an interactive dashboard.

---

## Live Demo

**Streamlit Application**

https://pearls-aqi-predictor-1.streamlit.app/

---

## GitHub Repository

https://github.com/Ahad-Khan-02/pearls-aqi-predictor

---

# Project Objectives

* Collect live AQI and weather data
* Generate engineered forecasting features
* Store features in Hopsworks Feature Store
* Train and evaluate multiple ML models
* Store trained models in Hopsworks Model Registry
* Automate feature and training pipelines
* Generate 72-hour AQI forecasts
* Explain model predictions using SHAP
* Provide AQI health alerts
* Deploy an interactive Streamlit dashboard

---

# System Architecture

```text
Open-Meteo API
      │
      ▼
Feature Pipeline
      │
      ▼
Feature Engineering
      │
      ▼
Hopsworks Feature Store
      │
      ▼
Training Pipeline
      │
      ▼
Model Evaluation
      │
      ▼
Model Registry
      │
      ▼
Streamlit Dashboard
      │
      ▼
Forecasting + Explainability
```

---

# Features

## Data Collection

* Open-Meteo Historical Data
* Open-Meteo Forecast Data
* AQI Pollutant Data

## Feature Engineering

### Time Features

* Hour
* Day
* Month
* Day of Week
* Weekend Indicator
* Rush Hour Indicator

### Lag Features

* previous_aqi
* aqi_lag_3
* aqi_lag_6
* aqi_lag_12

### Rolling Features

* rolling_avg_3
* rolling_avg_6
* rolling_avg_24

### Trend Features

* aqi_change
* aqi_trend

### Pollution Features

* pollution_index

---

# Machine Learning Models

Three machine learning models were trained and evaluated:

| Model            | MAE    | RMSE   | R²     |
| ---------------- | ------ | ------ | ------ |
| GradientBoosting | 1.8153 | 4.3660 | 0.9389 |
| RandomForest     | 1.9487 | 4.7165 | 0.9287 |
| Ridge Regression | 1.7785 | 5.0609 | 0.9179 |

### Best Model

Gradient Boosting Regressor

```text
R² Score = 0.9389
```

---

# Dashboard Modules

## Dashboard

* Live AQI Monitoring
* AQI Gauge
* Weather Snapshot
* Pollutant Monitoring
* Health Recommendations

## Forecast

* 72-Hour AQI Forecast
* Peak AQI Detection
* Best AQI Detection
* Hourly Forecast Breakdown

## Explainability

* SHAP Feature Importance
* Waterfall Analysis
* Top Feature Drivers
* Feature Impact Breakdown

## Model Analytics

* Model Leaderboard
* MAE Comparison
* RMSE Comparison
* R² Comparison
* Performance Radar

## Air Quality Insights

* AQI Distribution
* Hourly Trends
* Weekly Analysis
* Monthly Trends
* Correlation Matrix

---

# Explainable AI

SHAP (SHapley Additive Explanations) is used to explain model predictions.

Features include:

* Feature Importance Analysis
* Prediction Breakdown
* Top AQI Drivers
* Waterfall Visualizations

---

# AQI Alert System

The system automatically generates health alerts based on predicted AQI levels.

| AQI Range | Category                       |
| --------- | ------------------------------ |
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 300+      | Hazardous                      |

---

# MLOps Components

## Feature Store

Hopsworks Feature Store

* Centralized feature management
* Historical feature storage
* Online and offline access

## Model Registry

Hopsworks Model Registry

* Model versioning
* Model tracking
* Production model management

---

# CI/CD Automation

GitHub Actions automates the complete workflow.

## Feature Pipeline

Runs every hour:

* Fetch latest AQI data
* Generate features
* Update Feature Store

Workflow:

```text
.github/workflows/feature_pipeline.yml
```

## Training Pipeline

Runs daily:

* Load historical features
* Train models
* Evaluate performance
* Register best model

Workflow:

```text
.github/workflows/training_pipeline.yml
```

---

# Project Structure

```text
pearls-aqi-predictor
│
├── .github/workflows
│   ├── feature_pipeline.yml
│   └── training_pipeline.yml
│
├── app
│   ├── pages
│   ├── assets
│   └── streamlit_app.py
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── best_aqi_model.pkl
│   └── model_metrics.json
│
├── notebooks
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
│
├── src
│   ├── api
│   ├── data_pipeline
│   ├── features
│   ├── inference
│   ├── training
│   └── utils
│
├── requirements.txt
├── runtime.txt
├── render.yaml
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Ahad-Khan-02/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* Plotly
* Streamlit
* Hopsworks
* GitHub Actions
* Open-Meteo API

---

# License

This project is developed for the 10Pearls Shine Internship Program (Data Sciences) and is intended for educational and research purposes.

---

# Author

Ahad Khan
