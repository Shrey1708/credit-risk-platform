"""
Exploratory Data Analysis Script: Home Credit Default Risk
Standalone executable companion to eda.ipynb.
Usage: python notebooks/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root and src to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.utils import get_logger, Config, format_currency, format_percent, days_to_years
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger("eda_script")

def run_eda():
    logger.info("=" * 65)
    logger.info("Starting Exploratory Data Analysis Pipeline")
    logger.info("=" * 65)
    
    loader = DataLoader(data_dir=str(Config.DATA_DIR))
    
    # 1. Ingestion
    logger.info("1. Loading application train data...")
    app_train = loader.load_application_train(nrows=50000)
    logger.info(f"Loaded sample of {len(app_train):,} records with {len(app_train.columns)} features.")
    
    # 2. Target Distribution
    target_counts = app_train['TARGET'].value_counts()
    default_rate = app_train['TARGET'].mean() * 100
    logger.info(f"2. Target Imbalance: Repaid={target_counts[0]:,} ({(100-default_rate):.2f}%), "
                f"Defaulted={target_counts[1]:,} ({default_rate:.2f}%)")
    
    # 3. Missing Value Analysis
    missing_pct = (app_train.isnull().mean() * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    logger.info(f"3. Missing Values: {len(high_missing)} columns have >50% missing data.")
    logger.info(f"   Top missing: {', '.join(high_missing.head(5).index)}")
    
    # 4. External Bureau Scores
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in app_train.columns]
    logger.info("4. External Credit Bureau Score Telemetry:")
    for col in ext_cols:
        paid_m = app_train.loc[app_train['TARGET'] == 0, col].mean()
        def_m = app_train.loc[app_train['TARGET'] == 1, col].mean()
        corr = app_train[col].corr(app_train['TARGET'])
        logger.info(f"   - {col}: Mean(Repaid)={paid_m:.3f}, Mean(Default)={def_m:.3f}, Gap={(paid_m-def_m):.3f}, Corr={corr:.3f}")
        
    # 5. Age & Employment Demographics
    app_train['AGE_YEARS'] = days_to_years(app_train['DAYS_BIRTH'])
    anom_count = (app_train['DAYS_EMPLOYED'] == Config.PENSIONER_ANOM_VALUE).sum()
    logger.info("5. Demographics & Outliers:")
    logger.info(f"   - Average Applicant Age: {app_train['AGE_YEARS'].mean():.1f} years (Median: {app_train['AGE_YEARS'].median():.1f})")
    logger.info(f"   - Days Employed Anomaly (365243): {anom_count:,} records ({anom_count/len(app_train)*100:.2f}%)")
    
    # 6. Pipeline Validation with DataPreprocessor
    logger.info("6. Testing DataPreprocessor on sample...")
    preprocessor = DataPreprocessor(
        missing_threshold=Config.MISSING_THRESHOLD, 
        corr_threshold=Config.CORR_THRESHOLD
    )
    clean_df = preprocessor.preprocess(app_train.sample(5000, random_state=Config.RANDOM_STATE))
    
    logger.info("=" * 65)
    logger.info("EDA Pipeline Validation Summary:")
    logger.info(f" - Cleaned Data Shape: {clean_df.shape}")
    logger.info(f" - Total Missing Values: {clean_df.isnull().sum().sum()}")
    logger.info(f" - DAYS_EMPLOYED_ANOM Flag in output: {'DAYS_EMPLOYED_ANOM' in clean_df.columns}")
    logger.info(f" - Selected Features Count: {len(preprocessor.selected_features)}")
    logger.info("=" * 65)
    logger.info("EDA script completed successfully!")

if __name__ == '__main__':
    run_eda()
