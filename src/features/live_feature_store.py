import sys
import os


# Add project root to path so 'src' package is findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

from dotenv import load_dotenv
import hopsworks
import pandas as pd

# load env variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env file.")

 
# login to hopsworks and get feature store
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

 
# get feature store
fs = project.get_feature_store()

 
# load live featured dataset from csv
data_path = "data/processed/live_featured_aqi_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")

 
# create feature group in feature store
feature_group = fs.get_or_create_feature_group(
    name="aqi_live_features",
    version=1,
    description="AQI forecasting features using Open-Meteo historical data",
    primary_key=["timestamp"],
    event_time="timestamp"
)

 
# insert data into feature group
df = df.drop_duplicates(subset=["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna()

required_cols = [
    "timestamp",
    "aqi",
    "pm25",
    "temperature"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


feature_group.insert(
    df,
    write_options={
        "wait_for_job": True
    }
)

print("Feature Store upload completed successfully!")