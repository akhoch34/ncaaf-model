from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# Use absolute path to ensure consistency
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Find project root by looking for .git or requirements.txt
# Start from cfb_predictor directory and go up
_project_root = _current_dir
for _ in range(5):  # Maximum 5 levels up
    # Check if we've found project root markers
    if os.path.exists(os.path.join(_project_root, '.git')) or \
       os.path.exists(os.path.join(_project_root, 'requirements.txt')) or \
       os.path.exists(os.path.join(_project_root, 'weekly_update.py')):
        break
    parent = os.path.dirname(_project_root)
    if parent == _project_root:  # Reached filesystem root
        break
    _project_root = parent

# Default data location is always src/data relative to project root
_default_data_dir = os.path.join(_project_root, "src", "data")
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", _default_data_dir))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
