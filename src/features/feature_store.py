import os
import tempfile

# =========================
# WINDOWS TEMP DIR FIX
# Must be done BEFORE importing hopsworks
# =========================

temp_dir = os.path.abspath("tmp")

# Store originals BEFORE any patching
_original_mkdir    = os.mkdir
_original_makedirs = os.makedirs
_original_chmod    = os.chmod
_original_open     = os.open

# Create temp dir using original makedirs
_original_makedirs(temp_dir, exist_ok=True)

# Set all temp env vars
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"]   = temp_dir
os.environ["TMP"]    = temp_dir

# Patch tempfile
tempfile.tempdir = temp_dir
tempfile.gettempdir = lambda: temp_dir

def _redirect(path):
    """Normalize any /tmp-prefixed path to our local temp_dir."""
    if isinstance(path, str) and path.startswith("/tmp"):
        path = os.path.normpath(path.replace("/tmp", temp_dir, 1))
    return path

# Patch os.mkdir
def _patched_mkdir(path, mode=0o777):
    path = _redirect(path)
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            _patched_mkdir(parent, mode)
        if not os.path.exists(path):
            _original_mkdir(path, mode)

os.mkdir = _patched_mkdir

# Patch os.makedirs
def _patched_makedirs(path, mode=0o777, exist_ok=False):
    path = _redirect(path)
    _original_makedirs(path, mode=mode, exist_ok=exist_ok)

os.makedirs = _patched_makedirs

# Patch os.chmod — no-op for /tmp paths
def _patched_chmod(path, mode, **kwargs):
    if isinstance(path, str) and path.startswith("/tmp"):
        return
    try:
        _original_chmod(path, mode, **kwargs)
    except (NotImplementedError, OSError):
        pass

os.chmod = _patched_chmod

# Patch os.open — redirect /tmp paths and ensure parent dirs exist
def _patched_open(path, flags, mode=0o777, **kwargs):
    path = _redirect(path)
    # Ensure parent directory exists before opening
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        _original_makedirs(parent, exist_ok=True)
    return _original_open(path, flags, mode, **kwargs)

os.open = _patched_open

# =========================
# NOW SAFE TO IMPORT
# =========================

import pandas as pd
import hopsworks
from dotenv import load_dotenv

# =========================
# PATCH _makedirs_with_sticky_bit DIRECTLY
# =========================

try:
    import hopsworks_common.client.external as _ext
    import inspect

    _target_class = None
    for name, obj in inspect.getmembers(_ext, inspect.isclass):
        if hasattr(obj, "_makedirs_with_sticky_bit"):
            _target_class = obj
            print(f"[patch] Found target class: {name}")
            break

    if _target_class:
        # Inspect the real path hopsworks builds so we mirror it exactly
        # From traceback: /tmp\eu-west.cloud.hopsworks.ai\pearls_aqi_predictor_01\adkhan02\
        # That's: /tmp / host / project_name / user /
        # hopsworks builds this via self._get_jks_dir_path() or similar
        # We patch to just ensure the full subtree exists under our temp_dir

        def _patched_makedirs_with_sticky_bit(self):
            """Windows-safe: build the real cert directory and create it."""
            # Reconstruct the path hopsworks would build under /tmp
            # Pattern: /tmp/<host>/<project>/<user>/
            try:
                raw_dir = self._get_jks_dir_path()
            except AttributeError:
                # Fallback: build it manually from known attributes
                parts = [self._host]
                if hasattr(self, '_project_name') and self._project_name:
                    parts.append(self._project_name)
                if hasattr(self, '_username') and self._username:
                    parts.append(self._username)
                raw_dir = "/tmp/" + "/".join(parts)

            directory = _redirect(raw_dir)
            _original_makedirs(directory, exist_ok=True)
            # Skip os.chmod — not needed on Windows

        _target_class._makedirs_with_sticky_bit = _patched_makedirs_with_sticky_bit
        print(f"[patch] Successfully patched _makedirs_with_sticky_bit on {_target_class.__name__}")
    else:
        print("[patch] No class found — relying on os-level patches")

except Exception as e:
    print(f"[patch] Direct patch failed: {e} — relying on os-level patches")

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

data_path = "data/processed/featured_aqi_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)

print(f"Loaded {len(df)} rows from {data_path}")
print(f"Columns: {list(df.columns)}")

# =========================
# CONVERT TIMESTAMP
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# =========================
# CREATE FEATURE GROUP
# =========================

feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
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