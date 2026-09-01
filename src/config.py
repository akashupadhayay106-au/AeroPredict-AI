import os
from dotenv import load_dotenv

load_dotenv()

# Data configuration
DATA_DIR = os.getenv("DATA_PATH", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model configuration
MODEL_DIR = os.getenv("MODEL_PATH", "models/ml")
DL_MODEL_DIR = "models/dl"
PREPROCESSOR_DIR = "models/preprocessors"

# C-MAPSS specific config
INDEX_COLUMNS = ['unit_number', 'time_cycles']
SETTING_COLUMNS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]

# Modeling parameters
RUL_CAP = 125 # Common cap value for RUL prediction (helps models not over-predict high RULs)
SEQUENCE_LENGTH = 30 # Window size for sequence models

# Risk Engine Thresholds (Configurable)
RISK_THRESHOLDS = {
    "LOW": 50,      # RUL > 50 -> LOW
    "MEDIUM": 25,   # 25 < RUL <= 50 -> MEDIUM
    "HIGH": 10,     # 10 < RUL <= 25 -> HIGH
    # RUL <= 10 -> CRITICAL
}

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
