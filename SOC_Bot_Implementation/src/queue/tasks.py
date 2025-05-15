from datetime import datetime, timezone
import os
import redis
from urllib.parse import urlparse
from stable_baselines3 import PPO
import src.config
from src.celery_app import celery_app
from src.realtime.qradar_ingestion import RLDummyEnv, update_rl_agent_with_feedback
from src.models.model_training import load_model

# Parse Redis connection settings from environment variable
redis_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
parsed_url = urlparse(redis_url)
redis_host = parsed_url.hostname or "localhost"
redis_port = parsed_url.port or 6379
redis_db   = int(parsed_url.path.replace("/", "") or 0)

redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

@celery_app.task
def train_rl_agent_task(feedback, model_version):
    """
    Task that triggers RL agent training given feedback.
    Also logs statistics to Redis per priority and taxonomy.
    """
    dummy_env = RLDummyEnv(observation_dim=387)
    rl_agent = PPO.load(src.config.RL_AGENT_PATH, env=dummy_env, device=src.config.DEVICE)
    model = load_model(model_version)

    guessed_pri = feedback["guessed_priority"]
    guessed_tax = feedback["guessed_taxonomy"]
    correct_pri = feedback["correct_priority"]
    correct_tax = feedback["correct_taxonomy"]

    # Use UTC date to group stats
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # === Logging prediction correctness ===
    if guessed_pri == correct_pri:
        redis_client.incr(f"stats:{date}:priority:{correct_pri}:correct")
    else:
        redis_client.incr(f"stats:{date}:priority:{correct_pri}:incorrect")

    if guessed_tax == correct_tax:
        redis_client.incr(f"stats:{date}:taxonomy:{correct_tax}:correct")
    else:
        redis_client.incr(f"stats:{date}:taxonomy:{correct_tax}:incorrect")

    # === Aggregated stats by day, type and category ===
    redis_client.incr(f"stats")                                 # total alerts all time
    redis_client.incr(f"stats:{date}")                          # total alerts that day
    redis_client.incr(f"stats:priority")                        # total priority alerts all time
    redis_client.incr(f"stats:priority:{correct_pri}")          # total for this priority all time
    redis_client.incr(f"stats:{date}:priority")                 # total priority alerts that day
    redis_client.incr(f"stats:{date}:priority:{correct_pri}")   # total for this priority that day

    redis_client.incr(f"stats:taxonomy")                        # total taxonomy alerts all time
    redis_client.incr(f"stats:taxonomy:{correct_tax}")          # total for this taxonomy all time
    redis_client.incr(f"stats:{date}:taxonomy")                 # total taxonomy alerts that day
    redis_client.incr(f"stats:{date}:taxonomy:{correct_tax}")   # total for this taxonomy that day

    # === RL Training ===
    result_info = update_rl_agent_with_feedback(feedback, model, rl_agent)
    return result_info