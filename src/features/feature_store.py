import sys
import os



# import tempfile
# # Windows-safe temp directory
# tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

# os.makedirs(
#     tempfile.tempdir,
#     exist_ok=True
# )


# Add project root to path so 'src' package is findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.hopsworks_windows_patch import apply_hopsworks_patches
apply_hopsworks_patches()


from dotenv import load_dotenv
import hopsworks
import pandas as pd





# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env file.")

# =========================
# LOGIN
# =========================

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

# =========================
# GET FEATURE STORE
# =========================

fs = project.get_feature_store()

# =========================
# LOAD FEATURE DATA
# =========================

data_path = "data/processed/live_featured_aqi_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")

# =========================
# CREATE FEATURE GROUP
# =========================

feature_group = fs.get_or_create_feature_group(
    name="aqi_live_features",
    version=1,
    description="AQI forecasting features using Open-Meteo historical data",
    primary_key=["timestamp"],
    event_time="timestamp"
)

# =========================
# INSERT DATA
# =========================

feature_group.insert(df)

print("Feature Store upload completed successfully!")