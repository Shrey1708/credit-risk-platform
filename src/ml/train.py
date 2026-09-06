import os
import sys
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from flaml import AutoML
except ImportError:
    raise ImportError("FLAML is not installed. Please run 'pip install flaml xgboost lightgbm'")

# Add src to path to import data modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_models(data_dir: str, models_dir: str, sample_size: int = None):
    """
    Trains multiple Machine Learning models to predict default probability.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    logging.info("Initializing Data Loader...")
    loader = DataLoader(data_dir=data_dir)
    df = loader.get_merged_train_data()

    if sample_size and sample_size < len(df):
        logging.info(f"Sampling {sample_size} rows for faster training...")
        df = df.sample(sample_size, random_state=42)

    # 2. Preprocess Data
    logging.info("Preprocessing Data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)

    # Prepare X and y
    if 'TARGET' not in df_clean.columns:
        raise ValueError("TARGET column not found in data.")
        
    y = df_clean['TARGET']
    # Drop identifiers and target
    drop_cols = ['TARGET', 'SK_ID_CURR']
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

    # 3. Train-Test Split
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save test set for evaluation later
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    joblib.dump((X_test, y_test), test_data_path)
    logging.info(f"Saved test set to {test_data_path}")

    # 4. Define and Train AutoML Model
    logging.info("Initializing FLAML AutoML...")
    automl = AutoML()
    
    # 10 minutes budget (600 seconds), optimized for roc_auc
    automl_settings = {
        "time_budget": 600,
        "metric": 'roc_auc',
        "task": 'classification',
        "log_file_name": os.path.join(models_dir, 'automl.log'),
        "seed": 42,
        "n_jobs": -1,
        "eval_method": "cv",
        "n_splits": 3
    }
    
    logging.info("Starting AutoML training and hyperparameter search (budget: 600 seconds)...")
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    logging.info(f"AutoML Best Model: {automl.best_estimator}")
    logging.info(f"AutoML Best Hyperparameters: {automl.best_config}")
    
    # Save the best model
    model_path = os.path.join(models_dir, "AutoML_Best_Model.pkl")
    joblib.dump(automl, model_path)
    logging.info(f"Saved AutoML_Best_Model to {model_path}")

    # Save the fitted preprocessor so predict.py can use it
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Saved preprocessor to {preprocessor_path}")
    
    logging.info("Training complete!")

if __name__ == "__main__":
    # Adjust paths relative to script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    # Use sample_size to speed up local execution if dataset is massive (e.g. 50k rows)
    # Set to None to use full dataset.
    train_models(data_dir=data_dir, models_dir=models_dir, sample_size=50000)
