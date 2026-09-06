import os
import gc
import logging
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.helpers import reduce_mem_usage

logger = get_logger("data_loader")

class DataLoader:
    """
    DataLoader is responsible for loading the raw datasets and performing 
    initial merges and aggregations, utilizing caching for performance.
    """
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir if data_dir is not None else str(Config.DATA_DIR)

    def _resolve_path(self, filename: str) -> str:
        """Finds the dataset file using multiple search paths."""
        direct_path = os.path.join(self.data_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        # Check standard fallback locations
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)),
            os.path.abspath(os.path.join(os.getcwd(), 'data', filename)),
            os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename)),
            os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', filename))
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return direct_path

    def load_table(self, filename: str, nrows: int = None) -> pd.DataFrame:
        """Generic loader for any dataset table with memory optimization."""
        path = self._resolve_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {filename} in {self.data_dir} or candidate search paths.")
        logger.info(f"Loading {filename} from {path}...")
        df = pd.read_csv(path, nrows=nrows)
        return self._reduce_mem_usage(df)

    def load_application_train(self, nrows: int = None) -> pd.DataFrame:
        """Loads application_train.csv."""
        return self.load_table("application_train.csv", nrows=nrows)

    def load_application_test(self, nrows: int = None) -> pd.DataFrame:
        """Loads application_test.csv."""
        return self.load_table("application_test.csv", nrows=nrows)

    def load_bureau(self, nrows: int = None) -> pd.DataFrame:
        """Loads bureau.csv."""
        return self.load_table("bureau.csv", nrows=nrows)

    def load_bureau_balance(self, nrows: int = None) -> pd.DataFrame:
        """Loads bureau_balance.csv."""
        return self.load_table("bureau_balance.csv", nrows=nrows)

    def load_previous_application(self, nrows: int = None) -> pd.DataFrame:
        """Loads previous_application.csv."""
        return self.load_table("previous_application.csv", nrows=nrows)

    def load_pos_cash_balance(self, nrows: int = None) -> pd.DataFrame:
        """Loads POS_CASH_balance.csv."""
        return self.load_table("POS_CASH_balance.csv", nrows=nrows)

    def load_installments_payments(self, nrows: int = None) -> pd.DataFrame:
        """Loads installments_payments.csv."""
        return self.load_table("installments_payments.csv", nrows=nrows)

    def load_credit_card_balance(self, nrows: int = None) -> pd.DataFrame:
        """Loads credit_card_balance.csv."""
        return self.load_table("credit_card_balance.csv", nrows=nrows)
        
    def _reduce_mem_usage(self, df):
        """Delegates memory reduction to centralized helper utility."""
        return reduce_mem_usage(df, verbose=True)

    def _agg_numeric(self, df, group_var, df_name):
        """Aggregates numeric columns of a dataframe."""
        numeric_df = df.select_dtypes(include=['number'])
        if group_var not in numeric_df.columns:
            numeric_df[group_var] = df[group_var]
            
        agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum'])
        agg.columns = [f'{df_name}_{col[0]}_{col[1]}' for col in agg.columns.values]
        agg.reset_index(inplace=True)
        return agg

    def get_merged_train_data(self) -> pd.DataFrame:
        """
        Loads the main application train dataset and merges it with aggregated features 
        from ALL auxiliary tables. Implements caching via parquet to avoid recomputing.
        """
        cache_path = self._resolve_path('merged_data.parquet')
        if os.path.exists(cache_path):
            logger.info(f"Loading cached merged data from {cache_path}...")
            return pd.read_parquet(cache_path)

        logger.info("Cache not found. Aggregating all datasets from scratch...")
        
        # 1. Main Application Data
        app_train_path = self._resolve_path("application_train.csv")
        df = pd.read_csv(app_train_path)
        df = self._reduce_mem_usage(df)
        
        # 2. Bureau and Bureau Balance
        bureau_path = self._resolve_path("bureau.csv")
        bureau_bal_path = self._resolve_path("bureau_balance.csv")
        
        if os.path.exists(bureau_path):
            logger.info("Processing bureau and bureau_balance...")
            bureau = pd.read_csv(bureau_path)
            
            if os.path.exists(bureau_bal_path):
                bureau_bal = pd.read_csv(bureau_bal_path)
                # Ensure STATUS is numeric or encoded if we want to agg. We'll stick to basic counts for simplicity
                bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').size().reset_index(name='BUREAU_BAL_COUNT')
                bureau = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU', how='left')
                del bureau_bal, bureau_bal_agg
                gc.collect()

            bureau_agg = self._agg_numeric(bureau, 'SK_ID_CURR', 'BUREAU')
            df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            del bureau, bureau_agg
            gc.collect()

        # 3. Previous Applications
        prev_path = self._resolve_path("previous_application.csv")
        if os.path.exists(prev_path):
            logger.info("Processing previous_application...")
            prev = pd.read_csv(prev_path)
            prev_agg = self._agg_numeric(prev, 'SK_ID_CURR', 'PREV')
            df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
            del prev, prev_agg
            gc.collect()

        # 4. POS CASH Balance
        pos_path = self._resolve_path("POS_CASH_balance.csv")
        if os.path.exists(pos_path):
            logger.info("Processing POS_CASH_balance...")
            pos = pd.read_csv(pos_path)
            pos_agg = self._agg_numeric(pos, 'SK_ID_CURR', 'POS')
            df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
            del pos, pos_agg
            gc.collect()

        # 5. Installments Payments
        inst_path = self._resolve_path("installments_payments.csv")
        if os.path.exists(inst_path):
            logger.info("Processing installments_payments...")
            inst = pd.read_csv(inst_path)
            # Engineer a quick feature: delay in days
            inst['PAYMENT_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
            inst_agg = self._agg_numeric(inst, 'SK_ID_CURR', 'INSTAL')
            df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
            del inst, inst_agg
            gc.collect()

        # 6. Credit Card Balance
        cc_path = self._resolve_path("credit_card_balance.csv")
        if os.path.exists(cc_path):
            logger.info("Processing credit_card_balance...")
            cc = pd.read_csv(cc_path)
            cc_agg = self._agg_numeric(cc, 'SK_ID_CURR', 'CC')
            df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
            del cc, cc_agg
            gc.collect()

        logger.info(f"Final merged dataset shape: {df.shape}")
        
        # Save to parquet for lightning-fast loading next time
        logger.info(f"Saving merged data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)
        
        return df
