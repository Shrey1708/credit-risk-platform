import time
import functools
import numpy as np
import pandas as pd
from typing import Union, Callable, Any
from src.utils.logger import get_logger

logger = get_logger("helpers")

def timeit(func: Callable) -> Callable:
    """Decorator to measure and log execution duration of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Function '{func.__name__}' executed in {duration:.2f}s")
        return result
    return wrapper

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Iterates through all numeric columns of a dataframe and modifies data types
    to minimize memory footprint without losing numerical precision.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(f"Memory usage decreased by {reduction:.1f}% ({start_mem:.1f} MB -> {end_mem:.1f} MB)")
        
    return df

def days_to_years(days: Union[int, float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """Converts negative dataset days (e.g. DAYS_BIRTH, DAYS_EMPLOYED) to positive years."""
    if isinstance(days, pd.Series):
        return (days.abs() / 365.25).round(1)
    elif isinstance(days, np.ndarray):
        return np.round(np.abs(days) / 365.25, 1)
    else:
        return round(abs(days) / 365.25, 1) if days is not None else 0.0

def years_to_days(years: Union[int, float]) -> float:
    """Converts positive user years to negative days for model compatibility."""
    return -1.0 * float(years) * 365.25

def format_currency(amount: Union[int, float]) -> str:
    """Formats a number as a human-readable USD currency string."""
    if amount is None or pd.isna(amount):
        return "$0"
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"${amount:,.0f}"
    else:
        return f"${amount:.2f}"

def format_percent(val: Union[int, float], decimals: int = 1) -> str:
    """Formats a decimal probability (0.147) as a percentage string (14.7%)."""
    if val is None or pd.isna(val):
        return "0.0%"
    return f"{val * 100:.{decimals}f}%"

def classify_risk_band(prob: float, low_threshold: float = 0.30, high_threshold: float = 0.60) -> str:
    """Classifies a default probability into standardized credit risk bands."""
    if prob < low_threshold:
        return "Low"
    elif prob < high_threshold:
        return "Medium"
    else:
        return "High"

def calculate_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Computes standard credit risk domain ratios on raw application data."""
    df = df.copy()
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_RATIO'] = (df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)).clip(0, 1)
        
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_RATIO'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)).clip(0, 20)
        
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['PAYMENT_RATE'] = (df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)).clip(0, 1)
        
    if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
        df['OVER_FINANCING_RATIO'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)).clip(0.5, 2.5)
        
    return df
