import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration settings for the Credit Risk Platform."""
    
    # Project Paths
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    DATABASE_PATH = BASE_DIR / "credit_risk.db"
    
    # Dataset Files
    APPLICATION_TRAIN_FILE = DATA_DIR / "application_train.csv"
    APPLICATION_TEST_FILE = DATA_DIR / "application_test.csv"
    MERGED_DATA_PARQUET = DATA_DIR / "merged_data.parquet"
    BUREAU_FILE = DATA_DIR / "bureau.csv"
    PREVIOUS_APP_FILE = DATA_DIR / "previous_application.csv"
    INSTALLMENTS_FILE = DATA_DIR / "installments_payments.csv"
    
    # Gemini LLM & Talk-To-Data Settings
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
    RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", 15))
    RATE_LIMIT_RPD = int(os.environ.get("RATE_LIMIT_RPD", 500))
    
    # Credit Risk Classification Bands
    LOW_RISK_THRESHOLD = 0.30     # Prob < 30% -> Low Risk
    HIGH_RISK_THRESHOLD = 0.60    # Prob >= 60% -> High Risk
    DEFAULT_CALIBRATION_FACTOR = 0.50 # Calibrated risk normalization factor
    
    # Preprocessing & Feature Engineering
    TARGET_COL = "TARGET"
    MISSING_THRESHOLD = 0.80      # Drop features with >80% missing values
    CORR_THRESHOLD = 0.95         # Drop collinear features with r > 0.95
    PENSIONER_ANOM_VALUE = 365243 # Known DAYS_EMPLOYED anomaly flag
    
    # Training Configuration
    TRAIN_SAMPLE_SIZE = 50000     # Fast training sample size
    AUTOML_TIME_BUDGET = 600      # 10 minutes FLAML budget
    RANDOM_STATE = 42

# Module-level alias constants for direct import convenience
BASE_DIR = Config.BASE_DIR
DATA_DIR = Config.DATA_DIR
MODELS_DIR = Config.MODELS_DIR
DATABASE_PATH = Config.DATABASE_PATH
GEMINI_API_KEY = Config.GEMINI_API_KEY
GEMINI_MODEL_NAME = Config.GEMINI_MODEL_NAME
LOW_RISK_THRESHOLD = Config.LOW_RISK_THRESHOLD
HIGH_RISK_THRESHOLD = Config.HIGH_RISK_THRESHOLD
PENSIONER_ANOM_VALUE = Config.PENSIONER_ANOM_VALUE
