from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# Use absolute path to ensure consistency
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from cfb_predictor -> src -> project_root
_project_root = os.path.dirname(os.path.dirname(_current_dir))
# Default data location is src/data
_default_data_dir = os.path.join(_project_root, "src", "data")
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", _default_data_dir))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
