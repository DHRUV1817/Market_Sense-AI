# Data processing utilities
"""
Data processing utilities for MarketSense AI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Generator
from pathlib import Path
import re

def find_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the most likely date column in a DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Date column name or None if not found
    """
    # Check for columns with 'date' or 'time' in the name
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower()]
    
    if date_cols:
        return date_cols[0]
    
    # Check for columns that can be converted to datetime
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except:
            continue
    
    return None

def find_numeric_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Find numeric columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude
    
    Returns:
        List of numeric column names
    """
    exclude_cols = exclude_cols or []
    return [col for col in df.columns if col not in exclude_cols and 
            pd.api.types.is_numeric_dtype(df[col])]

def find_categorical_columns(df: pd.DataFrame, max_categories: int = 20) -> List[str]:
    """
    Find categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        max_categories: Maximum number of unique values to consider categorical
    
    Returns:
        List of categorical column names
    """
    categorical_cols = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or (
            pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= max_categories
        ):
            categorical_cols.append(col)
    
    return categorical_cols

def ensure_datetime_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensure a column is datetime type.
    
    Args:
        df: Input DataFrame
        date_col: Date column name
    
    Returns:
        DataFrame with converted date column
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # If conversion fails, return original DataFrame
            pass
    
    return df

def extract_date_components(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract date components from a datetime column.
    
    Args:
        df: Input DataFrame
        date_col: Datetime column name
    
    Returns:
        DataFrame with added date component columns
    """
    df = ensure_datetime_column(df, date_col)
    
    # Extract components if conversion was successful
    if pd.api.types.is_datetime64_dtype(df[date_col]):
        df['year'] = df[date_col].dt.year
        df['quarter'] = df[date_col].dt.quarter
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
    
    return df

def chunk_data(df: pd.DataFrame, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
    """
    Process a DataFrame in chunks to reduce memory usage.
    
    Args:
        df: Input DataFrame
        chunk_size: Number of rows per chunk
    
    Yields:
        DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def safe_division(numerator: Union[int, float], denominator: Union[int, float], default: Union[int, float] = 0) -> float:
    """
    Perform division with protection against zero division.
    
    Args:
        numerator: Division numerator
        denominator: Division denominator
        default: Default value to return if denominator is zero
    
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default

def normalize_text(text: str) -> str:
    """
    Normalize text for NLP processing.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()