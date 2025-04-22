# src/main.py

import os
import click
import redis
import uvicorn
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO

import asyncio
import src.config
from src.logger import get_logger
from src.preprocessing.preprocess import preprocess_bulk_alerts, clean_text
from src.models.model_training import load_model, train_rf_model, SentenceBertVectorizer
from src.realtime.qradar_ingestion import process_qradar_alert, RLDummyEnv
from src.queue.tasks import train_rl_agent_task

# Global variable to store the model once loaded/trained
model = None
model_v = None
logger = None
rl_agent = None
last_rl_mtime = os.path.getmtime(src.config.RL_AGENT_PATH)
training_lock = asyncio.Lock()
    
### CHANGED: Create a global SBERT vectorizer to avoid reinitializing it multiple times.
global_sbert_vectorizer = SentenceBertVectorizer(model_name="paraphrase-MiniLM-L6-v2")
from sentence_transformers import SentenceTransformer
global_sbert_vectorizer.model_ = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Define a Pydantic model for incoming alerts.
class Alert(BaseModel):
    description: str
    rule_name: str

class AlertFeedback(BaseModel):
    description: str
    rule_name: str
    correct_priority: str
    correct_taxonomy: str
    resolution: str

# Lifespan context manager for FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code can be added here if needed.
    yield
    # Shutdown: Save the RL agent if it exists.
    global rl_agent
    try:
        mtime = os.path.getmtime(src.config.RL_AGENT_PATH)
    except FileNotFoundError:
        mtime = None

    if mtime and mtime > last_rl_mtime:
        dummy_env = RLDummyEnv(observation_dim=386)
        rl_agent = PPO.load(src.config.RL_AGENT_PATH, env=dummy_env)
        last_rl_mtime = mtime
        logger.info(f"Detected updated RL agent on disk; reloaded from {src.config.RL_AGENT_PATH}")

    rl_agent.save(src.config.RL_AGENT_PATH)
    logger.info(f"RL agent saved to {src.config.RL_AGENT_PATH}")

# Create FastAPI app
app = FastAPI(title="SOC Bot API", description="API for handling QRadar alerts and returning model predictions.", lifespan=lifespan)

@app.get("/health")
async def health():
    logger.debug("Health check requested.")
    return {"status": "OK"}

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.get("/queue_length")
def queue_length():
    length = redis_client.llen("celery")  # Adjust key based on your configuration
    return {"queue_length": length}

@app.post("/alerts")
async def process_alert(alert: Alert):
    """
    Endpoint to receive an alert from QRadar SOAR.
    It processes the raw alert, applies preprocessing, and then uses the loaded model
    to return a prediction.
    """
    try:
        prediction = process_qradar_alert(model, alert)
        logger.info(f"Prediction: {prediction}")

        return prediction
    except Exception as e:
        logger.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/feedback")
async def process_feedback(feedback: AlertFeedback, background_tasks: BackgroundTasks):
    """
    Receive feedback and, only when appropriate, enqueue RL training.
    """
    # If feedback indicates low-priority or irrelevant taxonomy or a terminal resolution,
    # skip training.
    terminal_resolutions = {"Duplicate", "Done", "Declined", "Not Applicable/Not confirmed"}
    if (
        feedback.correct_priority == "P4"
        or feedback.correct_taxonomy == "other"
        or feedback.resolution in terminal_resolutions
    ):
        logger.info(
            f"Skipping training task (priority={feedback.correct_priority}, "
            f"taxonomy={feedback.correct_taxonomy}, resolution={feedback.resolution})"
        )
        return {"status": "Training skipped due to feedback conditions."}

    # Otherwise enqueue the task
    try:
        global model_v
        task = train_rl_agent_task.delay(feedback.model_dump(), model_v)
        logger.info(f"Enqueued training task {task.id}")
        return {"status": "Task queued", "task_id": task.id}
    except Exception as e:
        logger.error(f"Error enqueueing training task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/alerts/final")
async def process_alert_final(alert: Alert):
    """
    New endpoint to return the final prediction using RF+RL.
    The process is as follows:
      1. Normalize the description and append rule_name if necessary.
      2. Get the baseline RF prediction and compute scaling factors.
      3. Compute the SBERT embedding using SentenceBertVectorizer.
      4. Build an observation vector: [scaled_priority, scaled_taxonomy, embedding].
      5. If the global RL agent is available, use it to predict an action vector:
            - Map action[0] to adjusted Priority index (scale and round).
            - Map action[1] similarly for Taxonomy.
            - Action[2] threshold (>= 0.5) gives the FP decision.
            - Action[3] is the confidence score.
         If RL agent is not available, fall back to the RF model prediction.
    """
    try:
        logger.debug(f"Received MinimalAlert for final prediction: {alert.model_dump_json()}")

        global rl_agent, last_rl_mtime

        try:
            mtime = os.path.getmtime(src.config.RL_AGENT_PATH)
        except FileNotFoundError:
            mtime = None

        if mtime and mtime > last_rl_mtime:
            dummy_env = RLDummyEnv(observation_dim=386)
            rl_agent = PPO.load(src.config.RL_AGENT_PATH, env=dummy_env)
            last_rl_mtime = mtime
            logger.info(f"Detected updated RL agent on disk; reloaded from {src.config.RL_AGENT_PATH}")

        normalized_text = clean_text(alert.description)
        if alert.rule_name.lower() not in normalized_text.lower():
            normalized_text = f"{normalized_text} {alert.rule_name}"
        
        # Get baseline RF prediction.
        rf_pred = model["pipeline"].predict([normalized_text])[0]
        rf_probas = model["pipeline"].predict_proba([normalized_text])
        rf_pri_conf = float(rf_probas[0][0][rf_pred[0]])
        rf_tax_conf = float(rf_probas[1][0][rf_pred[1]])
        rf_confidence = (rf_pri_conf + rf_tax_conf) / 2.0

        num_priority = len(model["le_priority"].classes_)
        num_taxonomy = len(model["le_taxonomy"].classes_)
        scaled_priority = rf_pred[0] / (num_priority - 1) if num_priority > 1 else 0.0
        scaled_taxonomy = rf_pred[1] / (num_taxonomy - 1) if num_taxonomy > 1 else 0.0
        
        ### CHANGED: Use the global SBERT vectorizer instead of reinitializing.
        embedding = global_sbert_vectorizer.transform([normalized_text])[0]
        
        obs = np.concatenate(([scaled_priority, scaled_taxonomy], embedding)).astype(np.float32)
        
        if rl_agent is None:
            logger.info("RL agent not initialized; falling back to RF prediction.")
            priority_label = model["le_priority"].inverse_transform([rf_pred[0]])[0]
            taxonomy_label = model["le_taxonomy"].inverse_transform([rf_pred[1]])[0]
            return {"Priority": priority_label, "Taxonomy": taxonomy_label, "Is_FP": False, "Confidence": 0.0}
        
        action, _ = rl_agent.predict(obs, deterministic=True)

        confidence = float(action[3])
        logger.debug(f"RL action: {action} with confidence: {confidence}")

        if confidence < rf_confidence:
            final_priority_idx = rf_pred[0]
            final_taxonomy_idx = rf_pred[1]
            used = "RF"
            logger.info("RL confidence below RF probability; using RF predictions.")
        else:
            final_priority_idx = int(round(action[0] * (num_priority - 1)))
            final_taxonomy_idx = int(round(action[1] * (num_taxonomy - 1)))
            used = "RL"
            logger.info("Using RL adjusted predictions.")
                
        priority_label = model["le_priority"].inverse_transform([final_priority_idx])[0]
        taxonomy_label = model["le_taxonomy"].inverse_transform([final_taxonomy_idx])[0]
        
        return {
            "Priority": priority_label,
            "Taxonomy": taxonomy_label,
            "Is_FP": bool(action[2] >= 0.5),           
            "RL_conf": confidence,
            "RF_conf": rf_confidence,
            "Used": used,
        }
    except Exception as e:
        logger.error(f"Error processing final alert: {e}")
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
    global logger, model, rl_agent, model_v
    model_v = model_version
    src.config.VERBOSE = verbose
    logger = get_logger(__name__)

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
        
    logger.info("Model is ready for use. Loading RL agent...")

    dummy_env = RLDummyEnv(observation_dim=386)
    if os.path.exists(src.config.RL_AGENT_PATH):
        try:
            rl_agent = PPO.load(src.config.RL_AGENT_PATH, env=dummy_env)
            logger.info(f"Loaded RL agent from {src.config.RL_AGENT_PATH}")
        except Exception as e:
            logger.error(f"Error loading RL agent: {e}")
            rl_agent = PPO(
                "MlpPolicy",
                dummy_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.9,
                ent_coef=0.01,
                verbose=1,
            )
            logger.info("Initialized new RL agent due to load error.")
    else:
        rl_agent = PPO(
            "MlpPolicy",
            dummy_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.9,
            ent_coef=0.01,
            verbose=1,
        )
        logger.info("Initialized new RL agent as no saved agent was found.")

    logger.info("RF model and RL agent are ready. Starting API server...")
    
    # Start the FastAPI server using uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug" if verbose else "info")

if __name__ == "__main__":
    main()
