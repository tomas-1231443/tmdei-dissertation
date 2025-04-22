import numpy as np
import gym
from gym import spaces

import src.config
from src.logger import with_logger
from src.preprocessing.preprocess import clean_text
from src.models.model_training import predict_alert, SentenceBertVectorizer

from stable_baselines3 import PPO

### CHANGED: Define a custom dummy environment with correct observation and action spaces.
class RLDummyEnv(gym.Env):
    def __init__(self, observation_dim=386):  # 2 (scaled RF values) + 384 (SBERT embedding)
        super(RLDummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    def step(self, action):
        return self.reset(), 0, True, {}

@with_logger
def process_qradar_alert(model, alert, *, logger):
    logger.debug(f"Received Alert: {alert.model_dump_json()}")

    # 1. Normalize the description.
    normalized_text = clean_text(alert.description)
    logger.debug(f"Normalized text: {normalized_text}")

    # 2. Ensure rule name is present; if not, append it.
    if alert.rule_name.lower() not in normalized_text.lower():
        logger.debug("Rule name not found in normalized text; appending it.")
        normalized_text = f"{normalized_text} {alert.rule_name}"
    
    # 3. Generate a prediction using the loaded RF model.
    # Note: This uses the 'predict' method; adjust if your predict function differs.
    prediction = predict_alert(model, normalized_text)
    logger.debug(f"Prediction: {prediction}")

    return prediction

##############################
# Custom Gym Environment     #
##############################

class FeedbackEnv(gym.Env):
    """
    A custom Gym environment for RL training on a single alert feedback.
    
    Observation:
      A vector constructed as follows:
        - Two elements: scaled RF predictions for Priority and Taxonomy.
          (Scaled by dividing by (#classes - 1))
        - A feature vector from the alert description embedding.
    
    Action:
      A continuous vector of 4 elements in [0,1].
      * We only care about element index 2: if >= 0.5, the RL agent is predicting False Positive.
    
    Reward:
      +1 if the RL agent's FP prediction (action[2] thresholded) matches the ground truth label from feedback.
      -1 otherwise.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sample, rf_model, label_encoders, embedding_dim=384):
        """
        Args:
            sample (dict): A dictionary with keys:
                - description (str): Original alert description.
                - rule_name (str)
                - feedback_label (bool): Ground truth; True means False Positive, False means True Positive.
                  (We derive this from the resolution mapping.)
            rf_model: The RF model pipeline to obtain baseline predictions.
            label_encoders (tuple): (le_priority, le_taxonomy) for scaling the predictions.
            embedding_dim (int): Dimension of the embedding vector.
        """
        super().__init__()
        self.sample = sample
        self.rf_model = rf_model
        self.le_priority, self.le_taxonomy = label_encoders
        self.embedding_dim = embedding_dim

        # Determine number of classes for scaling.
        self.num_priority = len(self.le_priority.classes_)
        self.num_taxonomy = len(self.le_taxonomy.classes_)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + self.embedding_dim,), dtype=np.float32)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.done = False

        self.ground_truth_fp = sample["feedback_label"]

        self.sbert_vectorizer = SentenceBertVectorizer(model_name="paraphrase-MiniLM-L6-v2")

        from sentence_transformers import SentenceTransformer
        self.sbert_vectorizer.model_ = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    def _get_observation(self):
        """
        Constructs the observation vector.

        Returns:
            np.ndarray: A vector of length (2 + embedding_dim) where the first two elements
            are scaled RF predictions and the remaining elements are the SBERT embedding.
        """

        # Normalize the description using your clean_text function.
        normalized_text = clean_text(self.sample["description"])
        # Append the rule_name if not present (case-insensitive check).
        if self.sample["rule_name"].lower() not in normalized_text.lower():
            normalized_text = f"{normalized_text} {self.sample['rule_name']}"

        # Get baseline RF prediction.
        # Expecting the RF model's predict() to return a tuple like (priority, taxonomy).
        rf_pred = self.rf_model.predict([normalized_text])[0]
        scaled_priority = rf_pred[0] / (self.num_priority - 1) if self.num_priority > 1 else 0.0
        scaled_taxonomy = rf_pred[1] / (self.num_taxonomy - 1) if self.num_taxonomy > 1 else 0.0
        
        embedding = self.sbert_vectorizer.transform([normalized_text])[0]

        obs = np.concatenate(([scaled_priority, scaled_taxonomy], embedding))
        return obs.astype(np.float32)
    
    def reset(self):
        self.done = False
        return self._get_observation()
    
    def step(self, action):
        """
        Apply the RL agent's action and compute the reward.

        Args:
            action (np.ndarray): Action vector (4 continuous values).

        Returns:
            observation (np.ndarray): The observation after the step.
            reward (float): +1 if the RL prediction matches ground truth, -1 otherwise.
            done (bool): Whether the episode is finished.
            info (dict): Additional info (e.g., RL FP prediction).
        """
        # Interpret FP prediction: if the third element is >= 0.5, consider it as FP.
        # Determine RL FP decision and extract confidence.
        rl_fp = action[2] >= 0.5
        confidence = float(action[3])

        # FP component: +1 if RL FP decision is correct, -1 otherwise.
        fp_reward = confidence if (rl_fp == self.ground_truth_fp) else -confidence
        w_fn = 0.5
        w_fp = 1.5
        w_tp = 2

        if rl_fp and not self.ground_truth_fp:
            # you predicted FP but it was a TP
            fp_reward = -w_fn  # smallish penalty
        elif not rl_fp and self.ground_truth_fp:
            # you predicted TP but it was an FP
            fp_reward = -w_fp  # heavier penalty
        else:
            # you got it right
            fp_reward = +w_tp * confidence

        # Compute RL-adjusted priority and taxonomy indices from the action.
        adjusted_priority_index = int(round(action[0] * (self.num_priority - 1)))
        adjusted_taxonomy_index = int(round(action[1] * (self.num_taxonomy - 1)))

        true_priority_index = int(np.where(self.le_priority.classes_ == self.sample["correct_priority"])[0][0])
        true_taxonomy_index = int(np.where(self.le_taxonomy.classes_ == self.sample["correct_taxonomy"])[0][0])

        # Indicators: 0 if prediction is correct, 1 if incorrect.
        priority_error_indicator = 0 if adjusted_priority_index == true_priority_index else 1
        taxonomy_error_indicator = 0 if adjusted_taxonomy_index == true_taxonomy_index else 1

        # Fixed penalty weights: adjust these values as needed.
        alpha = 3   # penalty for an incorrect priority
        beta = 2    # penalty for an incorrect taxonomy
        error_penalty = alpha * priority_error_indicator + beta * taxonomy_error_indicator

        reward = fp_reward - error_penalty

        self.done = True  # Single-step episode.
        next_observation = self._get_observation()
        info = {
            "rl_fp": rl_fp,
            "ground_truth_fp": self.ground_truth_fp,
            "priority_error_indicator": priority_error_indicator,
            "taxonomy_error_indicator": taxonomy_error_indicator,
            "adjusted_priority_index": adjusted_priority_index,
            "adjusted_taxonomy_index": adjusted_taxonomy_index,
            "true_priority_index": true_priority_index,
            "true_taxonomy_index": true_taxonomy_index,
            "fp_reward": fp_reward,
            "confidence": confidence,
            "error_penalty": error_penalty,
            "action": action.tolist()  # making it JSON-serializable if needed
        }
        return next_observation, reward, self.done, info
    
    def render(self, mode="human"):
        print(f"FeedbackEnv: ground_truth_fp={self.ground_truth_fp}")


@with_logger
def update_rl_agent_with_feedback(feedback: dict, model, rl_agent, *, logger):

    logger.debug(f"Received feedback: {feedback}")
        
    # Map resolution to binary label.
    # For our purpose, if resolution contains "true positive", label as True Positive (i.e. ground_truth_fp=False).
    # Otherwise, label as False Positive (ground_truth_fp=True).
    if "true positive" in feedback["resolution"].lower():
        ground_truth_fp = False
    else:
        ground_truth_fp = True
    
    # Build a sample dict for the FeedbackEnv.
    sample = {
        "description": feedback["description"],
        "rule_name": feedback["rule_name"],
        # We include correct_priority and correct_taxonomy if needed;
        # for now the RL update focuses on FP classification.
        "correct_priority": feedback["correct_priority"],
        "correct_taxonomy": feedback["correct_taxonomy"],
        "feedback_label": ground_truth_fp  # This is the binary label we need.
    }
    
    logger.debug(f"Mapped feedback resolution to ground truth FP = {ground_truth_fp}")
    # Create a FeedbackEnv instance with the current sample.
    # We pass the global RF model and its label encoders (which are contained in 'model').
    le_priority = model["le_priority"]
    le_taxonomy = model["le_taxonomy"]

    ### CHANGED: Use embedding_dim=384 to match SBERT output.
    feedback_env = FeedbackEnv(sample, model["pipeline"], (le_priority, le_taxonomy), embedding_dim=384)
    obs = feedback_env.reset()
    
    # Use the RL agent to take an action on this sample.
    # If the RL agent is not initialized, create it with a dummy environment.

    if rl_agent is None:
        # Create a dummy environment for initialization.
        dummy_env = RLDummyEnv(observation_dim=2+384)
        rl_agent = PPO("MlpPolicy", dummy_env, verbose=1)
        logger.info("Initialized new RL agent (PPO) as it was None.")
    
    # Before training, get the RL agent's current action on the sample.
    current_action, _ = rl_agent.predict(obs, deterministic=True)
    current_fp_prediction = current_action[2] >= 0.5
    logger.info(f"Before training, RL predicted FP: {current_fp_prediction}")
    
    # Train the RL agent on this feedback sample.
    # We use the feedback environment and run a small number of timesteps.
    rl_agent.set_env(feedback_env)
    rl_agent.learn(total_timesteps=512, reset_num_timesteps=False)
    
    # After training, get the updated action.
    updated_action, _ = rl_agent.predict(obs, deterministic=True)
    updated_fp_prediction = updated_action[2] >= 0.5
    logger.info(f"After training, RL predicted FP: {updated_fp_prediction}")
    
    info = {
        "status": "Feedback processed and RL agent updated.",
        "rl_fp": bool(updated_action[2] >= 0.5),
        "ground_truth_fp": bool(ground_truth_fp),
        "priority_error_indicator": 0 if int(round(updated_action[0] * (len(model["le_priority"].classes_)-1))) == int(np.where(model["le_priority"].classes_ == sample["correct_priority"])[0][0]) else 1,
        "taxonomy_error_indicator": 0 if int(round(updated_action[1] * (len(model["le_taxonomy"].classes_)-1))) == int(np.where(model["le_taxonomy"].classes_ == sample["correct_taxonomy"])[0][0]) else 1,
        "adjusted_priority_index": int(round(updated_action[0] * (len(model["le_priority"].classes_)-1))),
        "adjusted_taxonomy_index": int(round(updated_action[1] * (len(model["le_taxonomy"].classes_)-1))),
        "true_priority_index": int(np.where(le_priority.classes_ == sample["correct_priority"])[0][0]),
        "true_taxonomy_index": int(np.where(le_taxonomy.classes_ == sample["correct_taxonomy"])[0][0]),
        "fp_reward": 1.0 if (updated_action[2] >= 0.5) == sample["feedback_label"] else -1.0,
        "confidence": float(updated_action[3]),
        "error_penalty": 3 * (0 if int(round(updated_action[0] * (len(model["le_priority"].classes_)-1))) == int(np.where(model["le_priority"].classes_ == sample["correct_priority"])[0][0]) else 1) + 2 * (0 if int(round(updated_action[1] * (len(model["le_taxonomy"].classes_)-1))) == int(np.where(model["le_taxonomy"].classes_ == sample["correct_taxonomy"])[0][0]) else 1),
        "action": list(map(float, updated_action.tolist()))
    }
    reward = (1 + float(updated_action[3])) * (1 if (updated_action[2] >= 0.5) == sample["feedback_label"] else -1.0) - info["error_penalty"]
    info["reward"] = float(reward)

    rl_agent.save(src.config.RL_AGENT_PATH)

    return info


