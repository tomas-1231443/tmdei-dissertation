import json
from src.preprocessing.preprocess import preprocess_single_alert

def process_qradar_alert(alert_json: str) -> dict:
    """
    Process an alert coming from QRadar in JSON format.
    
    This function converts the JSON alert to a dictionary with keys matching
    the expected Excel column names, preprocesses it, and returns the cleaned alert.
    
    Parameters:
        alert_json (str): A JSON string representing the alert.
    
    Returns:
        dict: Preprocessed alert data ready for further processing or model inference.
    """
    try:
        alert_dict = json.loads(alert_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    
    # TODO: If the QRadar alert keys differ from our expected format, add a mapping here.
    # For now, we assume the JSON keys match our Excel column names.
    
    preprocessed_alert = preprocess_single_alert(alert_dict)
    return preprocessed_alert
