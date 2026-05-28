import sys
import os

# Add project root to path so 'src' package is findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Apply on ALL platforms — Kafka SSL cert issue affects both Windows and Linux CI
from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# =========================
# LOAD ENV
# =========================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env file.")

# =========================
# LOGIN
# =========================

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# =========================
# GET FEATURE GROUP
# =========================

feature_group = fs.get_feature_group(name="aqi_training_features", version=1)
df = feature_group.read()
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Loaded dataset shape: {df.shape}")

# =========================
# FEATURES & TARGET
# =========================

features = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week","is_weekend", "is_rush_hour",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6","rolling_avg_24", "aqi_change","aqi_trend", "pollution_index"
]
target = "future_aqi"

# =========================
# TRAIN / TEST SPLIT
# =========================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# =========================
# TRAIN MODELS
# =========================

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Ridge": Ridge()
}

best_model      = None
best_r2         = float("-inf")
best_model_name = None
met=[]

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2   = r2_score(y_test, predictions)
    met.append(([name, mae, rmse, r2]))

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    if r2 > best_r2:
        best_r2         = r2
        best_model      = model
        best_model_name = name

print(f"\nBest model: {best_model_name} (R2={best_r2:.4f})")


# =========================
# SAVE MODEL METRICS
# =========================

import json

metrics = {
    "RandomForest": {
        "MAE": round(met[0][1], 4),
        "RMSE": round(met[0][2], 4),
        "R2": round(met[0][3], 4)
    },
    "Ridge": {
        "MAE": round(met[1][1], 4),
        "RMSE": round(met[1][2], 4),
        "R2": round(met[1][3], 4)
    },
    "best_model": best_model_name
}

os.makedirs("models", exist_ok=True)

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model metrics saved successfully!")



# =========================
# SAVE MODEL LOCALLY
# =========================

os.makedirs("models", exist_ok=True)
model_path = "models/best_aqi_model.pkl"
joblib.dump(best_model, model_path)

# =========================
# UPLOAD TO MODEL REGISTRY
# =========================

mr = project.get_model_registry()

aqi_model = mr.python.create_model(
    name="aqi_forecasting_model",
    metrics={"r2_score": best_r2},
    description="AQI forecasting model trained from Hopsworks Feature Store"
)

aqi_model.save(model_path)

print("Model uploaded to Hopsworks Model Registry!")