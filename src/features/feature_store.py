import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

from dotenv import load_dotenv
import hopsworks
import pandas as pd

# =========================
# LOAD ENV
# =========================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# =========================
# LOGIN
# =========================

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

# =========================
# LOAD HISTORICAL DATA
# =========================

data_path = "data/processed/featured_aqi_data.csv"

df = pd.read_csv(data_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.drop_duplicates(subset=["timestamp"])

df = df.sort_values("timestamp")

df = df.dropna()

print(f"Loaded {len(df)} rows")

# =========================
# CREATE FEATURE GROUP
# =========================

feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Historical AQI training features",
    primary_key=["timestamp"],
    event_time="timestamp"
)

# =========================
# INSERT DATA
# =========================

feature_group.insert(
    df,
    write_options={
        "wait_for_job": True
    }
)

print("Training Feature Store upload completed!")