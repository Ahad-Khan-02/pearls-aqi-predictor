import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/processed/featured_aqi_data.csv")

# =========================
# FEATURES
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

# Inputs
X = df[features]

# Target
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
# MODELS
# =========================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

# Store results
results = []

best_model = None
best_r2 = float("-inf")

# =========================
# TRAIN & EVALUATE
# =========================

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    # Save results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# =========================
# RESULTS TABLE
# =========================

results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")
print(results_df)

# =========================
# SAVE BEST MODEL
# =========================

joblib.dump(
    best_model,
    "models/best_aqi_model.pkl"
)

print(f"\nBest Model: {best_model_name}")