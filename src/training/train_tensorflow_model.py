import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Apply Hopsworks patches
from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import hopsworks
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found.")

# =========================
# LOGIN TO HOPSWORKS
# =========================

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

fs = project.get_feature_store()

# =========================
# LOAD FEATURE GROUP
# =========================

feature_group = fs.get_feature_group(
    name="aqi_features",
    version=1
)

df = feature_group.read()

print(f"Dataset Shape: {df.shape}")

# =========================
# FEATURES & TARGET
# =========================

features = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "pm25",
    "pm10",
    "co",
    "no2",
    "o3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "previous_aqi",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "rolling_avg_3",
    "rolling_avg_6",
    "aqi_change"
]

X = df[features]
y = df["future_aqi"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# BUILD TENSORFLOW MODEL
# =========================

model = tf.keras.Sequential([

    tf.keras.layers.Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(1)

])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test).flatten()

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\nTensorFlow Model Performance")

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

local_model_path = "models/tensorflow_aqi_model.keras"

model.save(local_model_path)

print(f"\nModel saved locally at: {local_model_path}")

# =========================
# UPLOAD TO MODEL REGISTRY
# =========================

mr = project.get_model_registry()

tf_model = mr.tensorflow.create_model(
    name="aqi_tensorflow_model",
    metrics={
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2
    },
    description="TensorFlow AQI forecasting model"
)

tf_model.save(local_model_path)

print("\nTensorFlow model uploaded to Hopsworks!")

# =========================
# CLOSE CONNECTION
# =========================

project.disconnect()