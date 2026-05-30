import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Must apply patches BEFORE hopsworks.login() — works on Windows and Linux CI

# apply_hopsworks_patches()
from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import hopsworks
import pandas as pd
from dotenv import load_dotenv

 
# load env variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env file.")

 
# login to hopsworks and get feature store
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

 
# get feature group
feature_group = fs.get_feature_group(name="aqi_live_features", version=1)

 
# read data from feature group
df = feature_group.read()

 
# get latest row of data and convert to dict for prediction
latest_row = df.sort_values(by="timestamp", ascending=False).iloc[0]
latest_data = latest_row.to_dict()
latest_data.pop("future_aqi", None)
latest_data.pop("timestamp", None)
latest_data.pop("aqi", None)  

print(latest_data)

def get_latest_features():
    return latest_data