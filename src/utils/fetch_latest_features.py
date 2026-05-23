import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()

import hopsworks
import pandas as pd
from dotenv import load_dotenv

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
# GET FEATURE GROUP
# =========================

feature_group = fs.get_feature_group(
    name="aqi_features",
    version=1
)

# =========================
# READ DATA
# =========================

df = feature_group.read()

# =========================
# GET LATEST ROW
# =========================

latest_row = df.sort_values(
    by="timestamp",
    ascending=False
).iloc[0]

# =========================
# CONVERT TO DICT
# =========================

latest_data = latest_row.to_dict()

# Remove target column if exists
latest_data.pop("future_aqi", None)

print(latest_data)


# Return latest data
def get_latest_features():
    return latest_data