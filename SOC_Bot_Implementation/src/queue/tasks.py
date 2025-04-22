# tasks.py

from stable_baselines3 import PPO
import src.config
from src.celery_app import celery_app
from src.realtime.qradar_ingestion import RLDummyEnv, update_rl_agent_with_feedback
from src.models.model_training import load_model
# update_rl_agent_with_feedback should encapsulate your RL training logic 
# that currently runs in your /alerts/feedback endpoint.

@celery_app.task
def train_rl_agent_task(feedback, model_version):
    """
    Task that triggers RL agent training given feedback.
    
    :param feedback_data: Dictionary containing feedback details.
    """
    # Call your existing training code; for example:

    dummy_env = RLDummyEnv(observation_dim=386)
    rl_agent = PPO.load(src.config.RL_AGENT_PATH, env=dummy_env)

    model = load_model(model_version)
    result_info = update_rl_agent_with_feedback(feedback, model, rl_agent)
    return result_info
