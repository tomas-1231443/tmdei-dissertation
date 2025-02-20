import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from src.logger import get_logger
from src.config import DEFAULT_MODEL_PATH  # you might want to add this in config.py

logger = get_logger(__name__)

MODEL_PATH = DEFAULT_MODEL_PATH if 'DEFAULT_MODEL_PATH' in globals() else "src/models/trained_model.joblib"

def train_false_positive_detector(df: pd.DataFrame, label_col: str = "Status") -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier to detect if an alert is a false positive.
    
    Note:
    - Currently, this function assumes that the input features are numeric.
    - TODO: Integrate a text processing pipeline (e.g., using TfidfVectorizer) to handle raw text input.
      This would involve combining the text from relevant columns (such as 'Description' and other text fields)
      and converting them into a numeric feature vector before training.
    """
    # For now, we assume that non-label columns are numeric.
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"False-Positive Detector Accuracy: {score:.2f}")

    # Save the trained model to disk
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    return model

def load_trained_model():
    """
    Loads the trained model from disk if it exists.
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
        return model
    else:
        logger.info("No pre-trained model found. Training a new model.")
        return None
