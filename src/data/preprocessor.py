import os
import logging
import pandas as pd
import numpy as np
import joblib

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("preprocessor")

class DataPreprocessor:
    """
    DataPreprocessor handles cleaning, encoding, scaling, and advanced feature selection.
    Must be fitted on training data before being used for inference.
    """
    def __init__(self, missing_threshold=Config.MISSING_THRESHOLD, corr_threshold=Config.CORR_THRESHOLD):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.imputation_values = {}
        self.features_to_drop = set()
        self.selected_features = []
        self.is_fitted = False

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles known dataset anomalies based on exploratory analysis."""
        df = df.copy()
        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == Config.PENSIONER_ANOM_VALUE
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({Config.PENSIONER_ANOM_VALUE: np.nan})
        return df

    def fit(self, df: pd.DataFrame, target_col: str = 'TARGET'):
        """Fits the preprocessor to learn imputation values and perform feature selection."""
        logger.info("Fitting preprocessor and performing advanced feature selection...")
        df = self.handle_outliers(df)
        
        # 1. High Missing Value Filter
        missing_ratios = df.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        self.features_to_drop.update([c for c in high_missing_cols if c != target_col and c != 'DAYS_EMPLOYED_ANOM'])
        logger.info(f"Dropping {len(high_missing_cols)} features with >{self.missing_threshold*100}% missing values.")

        # Temporarily drop for further analysis
        df_temp = df.drop(columns=[c for c in self.features_to_drop if c in df.columns])

        # 2. Learn Imputation Values
        for col in df_temp.columns:
            if df_temp[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    val = df_temp[col].median()
                    self.imputation_values[col] = 0.0 if pd.isna(val) else val
                else:
                    mode_series = df_temp[col].mode()
                    self.imputation_values[col] = mode_series[0] if not mode_series.empty else 'Unknown'

        # Apply imputation to temp for correlation
        for col, val in self.imputation_values.items():
            df_temp[col] = df_temp[col].fillna(val)

        # 3. Collinear Feature Filter (Correlation > 0.95)
        # Limit to numeric columns for correlation
        numeric_df = df_temp.select_dtypes(include=[np.number])
        # To save memory, sample if huge
        if len(numeric_df) > 50000:
            numeric_df = numeric_df.sample(50000, random_state=42)
            
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        collinear_cols = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        
        self.features_to_drop.update([c for c in collinear_cols if c != target_col and c != 'DAYS_EMPLOYED_ANOM'])
        logger.info(f"Dropping {len(collinear_cols)} highly collinear features.")

        # 4. Zero Variance Filter
        variances = numeric_df.var()
        zero_var_cols = variances[variances == 0].index.tolist()
        self.features_to_drop.update([c for c in zero_var_cols if c != target_col and c != 'DAYS_EMPLOYED_ANOM'])
        logger.info(f"Dropping {len(zero_var_cols)} zero-variance features.")

        self.is_fitted = True
        
        # We will determine selected features during transform after get_dummies
        logger.info(f"Total features marked for dropping: {len(self.features_to_drop)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies cleaning, dropping, imputing, and encoding based on fitted logic."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
            
        logger.info("Transforming data...")
        df = self.handle_outliers(df)
        
        # Drop features
        cols_to_drop = [c for c in self.features_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Impute
        for col, val in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                
        # Fill any remaining NAs (e.g. new inference columns) with 0 or mode
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                 if pd.api.types.is_numeric_dtype(df[col]):
                     df[col] = df[col].fillna(0)
                 else:
                     mode_series = df[col].mode()
                     df[col] = df[col].fillna(mode_series[0] if not mode_series.empty else 'Unknown')

        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Clean up column names for LightGBM/XGBoost compatibility (remove special JSON characters)
        import re
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
        
        # Ensure exact column match with what was seen during training (post-encoding)
        if self.selected_features:
            missing_cols = [col for col in self.selected_features if col not in df.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
                df = pd.concat([df, missing_df], axis=1)
            # Reorder and drop extras
            df = df[[c for c in self.selected_features if c in df.columns] + 
                    ([c for c in ['TARGET', 'SK_ID_CURR'] if c in df.columns])]
        else:
            # First transform run (during training), save the final features
            self.selected_features = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
            
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'TARGET') -> pd.DataFrame:
        self.fit(df, target_col)
        return self.transform(df)

    def preprocess(self, df: pd.DataFrame, target_col: str = 'TARGET') -> pd.DataFrame:
        """
        Convenience method to preprocess a dataset.
        If the preprocessor is not yet fitted, fits and transforms the data.
        If already fitted, transforms the data using learned parameters.
        """
        if not self.is_fitted:
            return self.fit_transform(df, target_col=target_col)
        return self.transform(df)
