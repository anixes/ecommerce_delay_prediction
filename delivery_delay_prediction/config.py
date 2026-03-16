import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Database Configuration

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "olist")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Feature Schema
CAT_FEATURES = [
    'customer_state', 
    'seller_state', 
    'product_category', 
    'primary_payment_type', 
    'purchase_month', 
    'purchase_day_of_week', 
    'purchase_hour'
]

# Model Filenames
CATBOOST_BASELINE_MODEL = MODELS_DIR / "catboost_baseline.cbm"
CATBOOST_TUNED_MODEL = MODELS_DIR / "catboost_tuned.cbm"
LIGHTGBM_BASELINE_MODEL = MODELS_DIR / "lightgbm_baseline.txt"
LIGHTGBM_TUNED_MODEL = MODELS_DIR / "lightgbm_tuned.txt"
