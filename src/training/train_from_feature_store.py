import sys
import os

# Add project root to path so 'src' package is findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Apply on ALL platforms — Kafka SSL cert issue affects both Windows and Linux CI
from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load env
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env file.")

 
# login to hopsworks and get feature store
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
 
# get featured dataset from feature store
feature_group = fs.get_feature_group(name="aqi_training_features", version=1)
df = feature_group.read()
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Loaded dataset shape: {df.shape}")

# feature columns and target variable
features = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week","is_weekend", "is_rush_hour",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6","rolling_avg_24", "aqi_change","aqi_trend", "pollution_index"
]
target = "future_aqi"

 
# train-test split (80-20)
 
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

 
# train and evaluate models
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Ridge": Ridge(),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

best_model      = None
best_r2         = float("-inf")
best_model_name = None
metrics = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2   = r2_score(y_test, predictions)
    metrics[name] = {
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
    }

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    if r2 > best_r2:
        best_r2         = r2
        best_model      = model
        best_model_name = name

print(f"\nBest model: {best_model_name} (R2={best_r2:.4f})")


 
# save model metrics to json file
import json

metrics["best_model"] = best_model_name
os.makedirs("models", exist_ok=True)
with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Model metrics saved successfully!")



# save best model to disk
os.makedirs("models", exist_ok=True)
model_path = "models/best_aqi_model.pkl"
joblib.dump(best_model, model_path)

 
# upload model to Hopsworks Model Registry with metrics
mr = project.get_model_registry()

aqi_model = mr.python.create_model(
    name="aqi_forecasting_model",
    metrics={"r2_score": best_r2},
    description="AQI forecasting model trained from Hopsworks Feature Store"
)
aqi_model.save(model_path)
print("Model uploaded to Hopsworks Model Registry!")