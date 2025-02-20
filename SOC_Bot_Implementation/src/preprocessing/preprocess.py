import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Union

from src.config import EXCEL_COLUMNS

# TODO: Further down the line, integrate a pipeline approach (scikit-learn Pipeline)
# TODO: Add advanced text cleaning or normalization if needed

def preprocess_single_alert(alert_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single alert (represented as a dictionary of column->value).
    Returns a cleaned/normalized dictionary that can be fed into a model.
    """
    # Example cleaning for text fields
    # Removing punctuation, lowercasing, etc. You can expand as needed.
    for col in alert_dict:
        if isinstance(alert_dict[col], str):    
            # basic text normalization
            cleaned = alert_dict[col].lower()
            cleaned = re.sub(r"[^\w\s]", "", cleaned)  # remove punctuation
            alert_dict[col] = cleaned.strip()

    # TODO: handle missing or null values
    # e.g. if any essential field is missing, we might set a default or raise an error

    # Possibly do label encoding for "Status", "Issue Type", etc.
    # This is a minimal placeholder; real approach may require scikit-learn's LabelEncoder
    # or a mapping dictionary you maintain.

    return alert_dict


def preprocess_bulk_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess an entire DataFrame of alerts.
    Applies the same transformations as preprocess_single_alert but to each row.
    """
    # Ensure columns are as expected
    missing_cols = set(EXCEL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in the input data: {missing_cols}")

    # Example transformation on each row
    for col in df.columns:
        if df[col].dtype == object:  # likely string
            df[col] = df[col].fillna("").astype(str).str.lower()
            # remove punctuation
            df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)
            df[col] = df[col].str.strip()

    # TODO: handle numeric columns if needed, e.g., fill missing values with median
    # df['Priority'] = df['Priority'].fillna(df['Priority'].median())

    return df
