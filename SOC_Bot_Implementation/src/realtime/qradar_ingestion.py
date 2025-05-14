import numpy as np
import gymnasium as gym
from gymnasium import spaces
import re
import pandas as pd

import src.config
from src.logger import with_logger
from src.preprocessing.preprocess import clean_text
from src.models.model_training import predict_alert, SentenceBertVectorizer
from stable_baselines3 import PPO

class RLDummyEnv(gym.Env):
    def __init__(self, observation_dim=387):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

@with_logger
def process_qradar_alert(model, alert, *, logger):
    logger.debug(f"Received Alert: {alert.model_dump_json()}")

    critical_flag = 1 if re.search(
        r'Related with Critical Asset:\s*\{color:red\}True',
        alert.description,
        flags=re.IGNORECASE
    ) else 0
    logger.debug(f"CriticalAsset flag: {critical_flag}")

    normalized_text = clean_text(alert.description)
    logger.debug(f"Normalized text: {normalized_text}")

    # 2. Ensure rule name is present; if not, append it.
    if alert.rule_name.lower() not in normalized_text.lower():
        normalized_text = f"{normalized_text} {alert.rule_name}"

    # 3. Generate a prediction using the loaded RF model.
    prediction = predict_alert(model, normalized_text, critical_flag=critical_flag)
    logger.debug(f"Prediction: {prediction}")

    return prediction

class FeedbackEnv(gym.Env):
    """
    Custom Gymnasium environment for one-step RL training on alert feedback.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, sample, rf_model, label_encoders, embedding_dim=384):
        super().__init__()
        self.sample = sample
        self.rf_model = rf_model
        self.le_priority, self.le_taxonomy = label_encoders
        self.embedding_dim = embedding_dim

        # build spaces
        self.num_priority = len(self.le_priority.classes_)
        self.num_taxonomy = len(self.le_taxonomy.classes_)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + self.embedding_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # feedback label
        self.ground_truth_fp = sample["feedback_label"]

        # SBERT
        self.sbert_vectorizer = SentenceBertVectorizer("paraphrase-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        self.sbert_vectorizer.model_ = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def _get_observation(self):
        critical_flag = 1 if re.search(
            r'Related with Critical Asset:\s*\{color:red\}True',
            self.sample["description"], flags=re.IGNORECASE
        ) else 0

        text = clean_text(self.sample["description"])
        if self.sample["rule_name"].lower() not in text.lower():
            text = f"{text} {self.sample['rule_name']}"

        X = pd.DataFrame([{
            "Description": text,
            "CriticalAsset": critical_flag
        }])

        rf_pred = self.rf_model.predict(X)[0]
        sp = rf_pred[0] / (self.num_priority - 1) if self.num_priority > 1 else 0.0
        st = rf_pred[1] / (self.num_taxonomy - 1) if self.num_taxonomy > 1 else 0.0
        emb = self.sbert_vectorizer.transform([text])[0]

        return np.concatenate(([sp, st, critical_flag], emb)).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.done = False
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # FP decision + confidence
        rl_fp = action[2] >= 0.5
        confidence = float(action[3])

        # simple reward: +confidence if correct, –confidence if wrong
        fp_reward = confidence if rl_fp == self.ground_truth_fp else -confidence

        # priority/taxonomy indices
        adj_pri = int(round(action[0] * (self.num_priority - 1)))
        adj_tax = int(round(action[1] * (self.num_taxonomy - 1)))
        true_pri = int(np.where(self.le_priority.classes_ == self.sample["correct_priority"])[0][0])
        true_tax = int(np.where(self.le_taxonomy.classes_ == self.sample["correct_taxonomy"])[0][0])

        # penalty for misclassification
        pri_err = 0 if adj_pri == true_pri else 1
        tax_err = 0 if adj_tax == true_tax else 1
        penalty = 3 * pri_err + 2 * tax_err

        reward = fp_reward - penalty

        self.done = True
        obs = self._get_observation()
        info = {
            "rl_fp": rl_fp,
            "ground_truth_fp": self.ground_truth_fp,
            "adjusted_priority_index": adj_pri,
            "adjusted_taxonomy_index": adj_tax,
            "true_priority_index": true_pri,
            "true_taxonomy_index": true_tax,
            "confidence": confidence,
            "fp_reward": fp_reward,
            "penalty": penalty,
            "action": action.tolist(),
        }
        return obs, reward, True, False, info

    def render(self, mode="human"):
        print(f"FeedbackEnv: ground_truth_fp={self.ground_truth_fp}")

@with_logger
def update_rl_agent_with_feedback(feedback: dict, model, rl_agent, *, logger):
    """
    Ingest one feedback sample, train RL agent for a few steps, and return metrics.
    """
    # map resolution→bool
    ground_truth_fp = not ("true positive" in feedback["resolution"].lower())

    sample = {
        "description": feedback["description"],
        "rule_name": feedback["rule_name"],
        "correct_priority": feedback["correct_priority"],
        "correct_taxonomy": feedback["correct_taxonomy"],
        "feedback_label": ground_truth_fp
    }

    env = FeedbackEnv(sample, model["pipeline"], (model["le_priority"], model["le_taxonomy"]), embedding_dim=384)
    obs, _ = env.reset()

    # ensure PPO has correct env
    rl_agent.set_env(env)
    rl_agent.learn(total_timesteps=10, reset_num_timesteps=False)

    new_action, _ = rl_agent.predict(obs, deterministic=True)
    info = {"updated_action": new_action.tolist()}
    rl_agent.save(src.config.RL_AGENT_PATH)
    return info