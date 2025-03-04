import os
import re
import glob
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.logger import get_logger
from src.config import DEFAULT_MODEL_DIR

logger = get_logger(__name__)

def get_model_filepath(version: int = None) -> str:
    """
    Returns the file path for the model of the specified version.
    If version is None, returns the filepath of the latest version.
    """
    if version is not None:
        return os.path.join(DEFAULT_MODEL_DIR, f"model_v{version}.joblib")
    else:
        files = glob.glob(os.path.join(DEFAULT_MODEL_DIR, "model_v*.joblib"))
        version_numbers = []
        for file in files:
            match = re.search(r"model_v(\d+)\.joblib", file)
            if match:
                version_numbers.append(int(match.group(1)))
        if version_numbers:
            latest_version = max(version_numbers)
            return os.path.join(DEFAULT_MODEL_DIR, f"model_v{latest_version}.joblib")
        else:
            return None
        
def get_next_version() -> int:
    """
    Returns the next version number for saving a new model.
    """
    files = glob.glob(os.path.join(DEFAULT_MODEL_DIR, "model_v*.joblib"))
    version_numbers = []
    for file in files:
        match = re.search(r"model_v(\d+)\.joblib", file)
        if match:
            version_numbers.append(int(match.group(1)))
    return max(version_numbers) + 1 if version_numbers else 1

def train_rf_model(df, label_col):
    """
    Trains a RandomForestClassifier on the provided DataFrame.
    Assumes that features are in numeric form. For raw text data, integrate a text
    vectorization pipeline (e.g., TfidfVectorizer) as a TODO.
    
    Returns the trained model and automatically saves it with a new version number.
    """

    model = None

    version = get_next_version()
    model_path = get_model_filepath(version)
    save_model(model, model_path)

    logger.info(f"Model version {version} saved to {model_path}")
    pass

def load_model(version: int = None) -> RandomForestClassifier:
    """
    Loads a trained model from disk.
    If version is provided, loads that specific version; otherwise, loads the most recent version.
    Returns None if no model file is found.
    """
    model_path = get_model_filepath(version)
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    else:
        logger.error("No model file found.")
        return None
    
def save_model(model: RandomForestClassifier, path: str) -> None:
    """
    Saves the trained model to disk at the given path.
    """
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
