import sys
import os

# Add project root to path so 'src' package is findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Apply on ALL platforms — Kafka SSL cert issue affects both Windows and Linux CI
from src.utils.hopsworks_windows_patch import apply_hopsworks_patches

import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

apply_hopsworks_patches()

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

feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()

print(f"Loaded dataset shape: {df.shape}")

# =========================
# FEATURES & TARGET
# =========================

features = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6", "aqi_change"
]

X = df[features]
y = df["future_aqi"]

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2   = r2_score(y_test, predictions)

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    if r2 > best_r2:
        best_r2         = r2
        best_model      = model
        best_model_name = name

print(f"\nBest model: {best_model_name} (R2={best_r2:.4f})")

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