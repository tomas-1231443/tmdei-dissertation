# src/main.py

import os
import json
import click
import uvicorn
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import src.config
from src.logger import get_logger
from src.preprocessing.preprocess import preprocess_bulk_alerts
from src.models.model_training import load_model, train_rf_model
from src.realtime.qradar_ingestion import process_qradar_alert

# Global variable to store the model once loaded/trained
model = None
logger = None

# Define a Pydantic model for incoming alerts.
class Alert(BaseModel):
    Issue_ID: str
    Issue_Type: str
    Status: str
    Description: str
    Custom_field_Alert_Technology: str
    Custom_field_Incident_Description: str
    Custom_field_Incident_Resolution_1: str
    Custom_field_Request_Type: str
    Custom_field_Source: str
    Custom_field_Source_Alert_Rule_Name: str
    Custom_field_Source_Alert_Id: str
    Custom_field_Taxonomy: str
    Priority: str

# Create FastAPI app
app = FastAPI(title="SOC Bot API", description="API for handling QRadar alerts and returning model predictions.")

@app.get("/health")
async def health():
    logger.debug("Health check requested.")
    return {"status": "OK"}

@app.post("/alerts")
async def process_alert(alert: Alert):
    """
    Endpoint to receive an alert from QRadar SOAR.
    It processes the raw alert, applies preprocessing, and then uses the loaded model
    to return a prediction.
    """
    try:
        # Convert Pydantic alert to a dictionary in the expected format.
        alert_dict = {
            "Issue ID": alert.Issue_ID,
            "Issue Type": alert.Issue_Type,
            "Status": alert.Status,
            "Description": alert.Description,
            "Custom field (Alert Technology)": alert.Custom_field_Alert_Technology,
            "Custom field (Incident Description)": alert.Custom_field_Incident_Description,
            "Custom field (Incident Resolution 1)": alert.Custom_field_Incident_Resolution_1,
            "Custom field (Request Type)": alert.Custom_field_Request_Type,
            "Custom field (Source)": alert.Custom_field_Source,
            "Custom field (Source Alert Rule Name)": alert.Custom_field_Source_Alert_Rule_Name,
            "Custom field (Source_Alert_Id)": alert.Custom_field_Source_Alert_Id,
            "Custom field (Taxonomy)": alert.Custom_field_Taxonomy,
            "Priority": alert.Priority
        }
        alert_json = json.dumps(alert_dict)
        logger.debug(f"Received alert: {alert_json}")
        
        # Process the alert using the QRadar ingestion function.
        processed_alert = process_qradar_alert(alert_json)
        logger.debug(f"Processed alert: {processed_alert}")

        # Convert the processed alert into a DataFrame for prediction.
        # Note: The model expects data in the same format as training.
        df = pd.DataFrame([processed_alert])
        prediction = model.predict(df)
        logger.info(f"Prediction for alert {alert_dict.get('Issue ID')}: {prediction[0]}")
        
        return {"Issue ID": alert_dict.get("Issue ID"), "Prediction": str(prediction[0])}
    except Exception as e:
        logger.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def preprocess(path):
    df = pd.read_csv(path)
    logger.debug(f"Loaded CSV data with shape: {df.shape}")
    df_clean = preprocess_bulk_alerts(df)
    df_clean.to_csv(src.config.DEFAULT_CLEANED_ALERTS_PATH, index=False)
    logger.debug("Preprocessing complete for historical data.")
    return df_clean
    
def train(path: str, tune=False) -> Optional[RandomForestClassifier]:

    if os.path.exists(src.config.DEFAULT_CLEANED_ALERTS_PATH):
        logger.info("Found preprocessed data. Loading cleaned CSV.")
        try:
            df_clean = pd.read_csv(src.config.DEFAULT_CLEANED_ALERTS_PATH)
            logger.debug(f"Loaded cleaned CSV data with shape: {df_clean.shape}")
            return train_rf_model(df_clean, tune)
        except Exception as e:
            logger.error(f"Error loading cleaned CSV data: {e}")
            return None

    if not os.path.exists(path):
        logger.error(f"Excel file not found at {path}. Aborting training.")
        return
    try:
        df_clean = preprocess(path)
    except Exception as e:
        logger.error(f"Error loading or preprocessing Excel data: {e}")
        return
    return train_rf_model(df_clean, tune)

@click.command()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging.")
@click.option("--excel-path", default=src.config.DEFAULT_EXCEL_PATH, help="Path to the Excel file with historical alerts.")
@click.option("--model-version", default=None, type=int, help="Specific model version to load. If not provided, loads the latest version.")
@click.option("--retrain", is_flag=True, default=False, help="Retrain the model and overwrite the existing one.")
@click.option("--tune", is_flag=True, default=False, help="Find the best set of parameters for the model for the current pre-processing.")
@click.option("--preprocess-only", is_flag=True, default=False, help="Preprocess the historical data only without training the model.")
@click.option("--port", default=8000, type=int, help="Port to run the API server on.")
def main(verbose, excel_path, retrain, model_version, port, preprocess_only, tune):
    """
    Main entry point that sets up the model (either by loading or training) and starts the API server.
    """
    global logger
    src.config.VERBOSE = verbose
    logger = get_logger(__name__)

    global model
    logger.info("Starting SOC Bot API Server")
    logger.debug(f"Arguments received - verbose: {verbose}, excel_path: {excel_path}, tune: {tune}, preprocess_only: {preprocess_only}, retrain: {retrain}, model_version: {model_version}, port: {port}")

    if preprocess_only:
        logger.info("Preprocess-only flag set. Preprocessing the historical data and exiting.")
        preprocess(excel_path)
        return

    # Model loading/training logic.
    if retrain:
        logger.info("Retrain flag set. Training a new model from historical data.")
        if tune:
            logger.info("Tune flag set. Finding best parameters for the current pre-processing method and exiting.")
            return train(excel_path, tune)
        model = train(excel_path)
    else:
        logger.info("Attempting to load existing trained model.")
        model = load_model(model_version)
        if model is None:
            logger.info("No existing model found. Training a new model.")
            model = train(excel_path)
    
    logger.info("Model is ready for use. Starting API server...")
    
    # Start the FastAPI server using uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug" if verbose else "info")

if __name__ == "__main__":
    main()
