import os
import re
import glob
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from src.logger import with_logger
from src.config import DEFAULT_MODEL_DIR


def get_model_filepath(version: int = None) -> Optional[str]:
    """
    Returns the file path for the model of the specified version.
    If version is None, returns the file path for the latest version model.
    The model is stored inside a folder named "V{version}".
    """
    if version is not None:
        version_folder = os.path.join(DEFAULT_MODEL_DIR, f"V{version}")
        model_path = os.path.join(version_folder, f"model_v{version}.joblib")
        return model_path
    else:
        # Find all folders matching "V<number>" in DEFAULT_MODEL_DIR.
        if not os.path.exists(DEFAULT_MODEL_DIR):
            return None
        dirs = [
            d for d in os.listdir(DEFAULT_MODEL_DIR)
            if os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, d)) and re.match(r"V\d+", d)
        ]
        version_numbers = []
        for d in dirs:
            match = re.match(r"V(\d+)", d)
            if match:
                version_numbers.append(int(match.group(1)))
        if version_numbers:
            latest_version = max(version_numbers)
            version_folder = os.path.join(DEFAULT_MODEL_DIR, f"V{latest_version}")
            model_path = os.path.join(version_folder, f"model_v{latest_version}.joblib")
            return model_path
        else:
            return None
        
def get_next_version() -> int:
    """
    Returns the next version number for saving a new model based on existing version folders.
    """
    if not os.path.exists(DEFAULT_MODEL_DIR):
        os.makedirs(DEFAULT_MODEL_DIR)
    dirs = [
        d for d in os.listdir(DEFAULT_MODEL_DIR)
        if os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, d)) and re.match(r"V\d+", d)
    ]
    version_numbers = []
    for d in dirs:
        match = re.match(r"V(\d+)", d)
        if match:
            version_numbers.append(int(match.group(1)))
    return max(version_numbers) + 1 if version_numbers else 1

@with_logger
def train_rf_model(df: pd.DataFrame, tune: bool = False, *, logger) -> Dict[str, Any]:
    """
    Trains a RandomForest model to predict both Priority and Taxonomy from the 'Description' text.
    
    This function:
      - Vectorizes the 'Description' column using TfidfVectorizer.
      - Encodes the 'Priority' and 'Taxonomy' targets using LabelEncoder.
      - Splits the data 60% for training and 40% for testing.
      - Trains a MultiOutputClassifier (with RandomForestClassifier as the base estimator).
      - Logs test set accuracy and classification reports.
    
    Returns:
        A dictionary containing:
          - "pipeline": A Pipeline that vectorizes raw text and outputs predictions.
          - "le_priority": The LabelEncoder used for the Priority labels.
          - "le_taxonomy": The LabelEncoder used for the Taxonomy labels.
    """
    logger.info("Starting training of the Random Forest model for multi-output classification.")

    # 1. Vectorize the Description column.
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df["Description"])    
    # 2. Encode targets.
    le_priority = LabelEncoder()
    le_taxonomy = LabelEncoder()
    y_priority = le_priority.fit_transform(df["Priority"])
    y_taxonomy = le_taxonomy.fit_transform(df["Taxonomy"])
    
    # Combine the two targets into a single 2D array.
    Y = np.column_stack((y_priority, y_taxonomy))
    logger.debug("Target labels encoded.")

    if tune:
        results = _tune_hyperparameters(df["Description"], Y)
        pipeline = results["pipeline"]
        logger.info(f"Best parameters found: {results['best_params']}, Best cross-validation score: {results['best_score']:.4f}")

        logger.info("Tunning complete.")
        
        trained_model = {
            "pipeline": pipeline,
            "le_priority": le_priority,
            "le_taxonomy": le_taxonomy
        }

        save_model(pipeline, get_next_version(), "Tuning results:\n" + f"Best parameters found: {results['best_params']}, Best cross-validation score: {results['best_score']:.4f}")

        return trained_model
    
    stratify_labels = [tuple(row) for row in Y]
    # 3. Split data: 70% train, 30% test best of all sets.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=stratify_labels)
    logger.debug(f"Data split into training and testing sets. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 4. Create and train a multi-output classifier.
    # Best set so far is 100 estimators, max depth of None, min samples split of 2, min samples leaf of 1, class weight = None
    base_rf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=40, min_samples_split=10, min_samples_leaf=1, class_weight=None)
    multi_rf = MultiOutputClassifier(base_rf)

    logger.info("Training started...")
    start_time = time.time()
    multi_rf.fit(X_train, Y_train)
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 5. Evaluate on the test set.
    Y_pred = multi_rf.predict(X_test)
    acc_priority = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
    acc_taxonomy = accuracy_score(Y_test[:, 1], Y_pred[:, 1])
    logger.info(f"Test Accuracy - Priority: {acc_priority:.2f}, Taxonomy: {acc_taxonomy:.2f}")
    logger.debug("Classification Report for Priority:\n" + classification_report(Y_test[:, 0], Y_pred[:, 0]))
    logger.debug("Classification Report for Taxonomy:\n" + classification_report(Y_test[:, 1], Y_pred[:, 1]))

    try:
        # Priority confusion matrix.
        logger.debug("Computing confusion matrix for Priority.")
        cm_priority = confusion_matrix(Y_test[:, 0], Y_pred[:, 0])
        cm_priority_norm = cm_priority.astype('float') / cm_priority.sum(axis=1)[:, np.newaxis]
        priority_class_names = le_priority.classes_
        logger.info("Priority confusion matrix computed and normalized.")

        logger.debug("Plotting Priority confusion matrix.")
        fig_priority = plt.figure(figsize=(10, 8))
        sns.set_theme(font_scale=1.2)
        sns.heatmap(cm_priority_norm, annot=True, cmap=plt.cm.Greens, fmt=".2f",
                    xticklabels=priority_class_names, yticklabels=priority_class_names, linewidths=0.5)
        plt.xlabel('Predicted Priority')
        plt.ylabel('True Priority')
        plt.title('Normalized Confusion Matrix for Priority')
        logger.info("Priority confusion matrix plot created.")
        
        # Taxonomy confusion matrix.
        logger.debug("Plotting Taxonomy confusion matrix.")
        cm_taxonomy = confusion_matrix(Y_test[:, 1], Y_pred[:, 1])
        cm_taxonomy_norm = cm_taxonomy.astype('float') / cm_taxonomy.sum(axis=1)[:, np.newaxis]
        taxonomy_class_names = le_taxonomy.classes_
        
        fig_taxonomy = plt.figure(figsize=(17, 15))
        sns.set_theme(font_scale=1.2)
        sns.heatmap(cm_taxonomy_norm, annot=True, cmap=plt.cm.Greens, fmt=".2f",
                    xticklabels=taxonomy_class_names, yticklabels=taxonomy_class_names, linewidths=0.5)
        plt.xlabel('Predicted Taxonomy')
        plt.ylabel('True Taxonomy')
        plt.title('Normalized Confusion Matrix for Taxonomy')
        logger.info("Taxonomy confusion matrix plot created.")
    except Exception as e:
        logger.error(f"Error while plotting confusion matrices: {e}")
        fig_priority, fig_taxonomy = None, None
    
    # 6. Create a pipeline that includes the vectorizer and the classifier.
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", multi_rf)
    ])
    
    # Return the pipeline and label encoders for later use in prediction.
    trained_model = {
        "pipeline": pipeline,
        "le_priority": le_priority,
        "le_taxonomy": le_taxonomy
    }
    
    version = get_next_version()
    report = f"Test Accuracy - Priority: {acc_priority:.2f}, Taxonomy: {acc_taxonomy:.2f}\n"
    report += "Classification Report for Priority:\n" + classification_report(Y_test[:, 0], Y_pred[:, 0])
    report += "\nClassification Report for Taxonomy:\n" + classification_report(Y_test[:, 1], Y_pred[:, 1])

    save_model(pipeline, version, report, priority_fig=fig_priority, taxonomy_fig=fig_taxonomy)

    logger.info("Training complete.")

    return trained_model

@with_logger
def load_model(version: int = None, *, logger) -> Optional[Pipeline]:
    """
    Loads a trained model pipeline from disk.
    
    Parameters:
        version (int, optional): Specific model version to load. If None, loads the latest version.
        logger: Logger instance for logging.
        
    Returns:
        Optional[Pipeline]: The loaded model pipeline if found, otherwise None.
    """
    model_path = get_model_filepath(version)
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    else:
        logger.error("No model file found.")
        return None
    
@with_logger
def save_model(model: Pipeline, version: int, report: str, priority_fig, taxonomy_fig, *, logger) -> None:
    """
    Saves the trained model into a folder named "V{version}" under DEFAULT_MODEL_DIR.
    Inside that folder, the model is saved as "model_v{version}.joblib" and a report is saved
    as "report.txt" containing evaluation metrics (e.g., average scores and classification reports).
    
    Parameters:
        model (Pipeline): The trained model pipeline.
        version (int): The version number for this model.
        report (str): A text string containing the evaluation report.
        logger: Logger instance for logging.
    """
    version_folder = os.path.join(DEFAULT_MODEL_DIR, f"V{version}")
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)
    model_path = os.path.join(version_folder, f"model_v{version}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    report_path = os.path.join(version_folder, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    priority_cm_path = os.path.join(version_folder, "confusion_priority.png")
    taxonomy_cm_path = os.path.join(version_folder, "confusion_taxonomy.png")
    priority_fig.savefig(priority_cm_path)
    taxonomy_fig.savefig(taxonomy_cm_path)

    logger.info(f"Confusion matrix for Priority saved to {priority_cm_path}")
    logger.info(f"Confusion matrix for Taxonomy saved to {taxonomy_cm_path}")

@with_logger
def multioutput_accuracy(y_true, y_pred, *, logger):
    """
    Computes the average accuracy for a multi-output classification problem.
    For each output, it computes the proportion of correctly predicted samples,
    then returns the average across outputs.
    """
    try:
        accuracies = []
        for i in range(y_true.shape[1]):
            acc = np.mean(y_true[:, i] == y_pred[:, i])
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        return avg_acc
    except Exception as e:
        logger.error(f"Error in multioutput_accuracy: {e}")
        raise e
    
scorer = make_scorer(multioutput_accuracy)

@with_logger
def _tune_hyperparameters(X: np.array, Y: np.array, *, logger) -> dict:
    """
    Tune hyperparameters for the text classification pipeline using GridSearchCV.

    Parameters:
        X (np.array): The raw text data (e.g. cleaned 'Description' column).
        Y (np.array): The multi-output target array (e.g. [priority, taxonomy]).

    Returns:
        dict: A dictionary containing:
            - "pipeline": The best pipeline found.
            - "best_params": The best parameter combination.
            - "best_score": The best cross-validation score.
    """
    logger.info("Starting hyperparameter tuning.")
    
    # Log the shape of the input data.
    logger.debug(f"Input X shape: {X.shape}")
    logger.debug(f"Input Y shape: {Y.shape}")
    
    # Create the pipeline with a TfidfVectorizer and a MultiOutputClassifier wrapping a RandomForestClassifier.
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=10000)),
        ("classifier", MultiOutputClassifier(RandomForestClassifier(random_state=42, verbose=0)))
    ])
    
    # Define the hyperparameter grid.
    param_grid = {
        'classifier__estimator__n_estimators': [25, 50, 75, 150, 200],
        'classifier__estimator__max_depth': [10, 20, 30, 40],
        'classifier__estimator__min_samples_split': [2, 5, 10],
        'classifier__estimator__min_samples_leaf': [1, 2, 4],
        'classifier__estimator__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    logger.debug(f"Parameter grid: {param_grid}")
    
    # Use KFold cross-validation (5 folds).
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    logger.info("Using KFold with 5 splits for cross-validation.")
    
    # Initialize GridSearchCV.
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    
    try:
        logger.info("Starting GridSearchCV.fit()")
        grid_search.fit(X, Y)
        logger.info("GridSearchCV.fit() completed successfully.")
    except Exception as e:
        logger.error(f"Error during GridSearchCV.fit(): {e}")
        raise e
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.debug(f"Best parameters found: {best_params}")
    logger.debug(f"Best cross-validation score: {best_score:.4f}")
    
    return {"pipeline": best_model, "best_params": best_params, "best_score": best_score}

    
def predict_alert(trained_model: dict, description: str) -> dict:
    """
    Uses the trained model pipeline and label encoders to predict Priority and Taxonomy from a raw description.
    
    Parameters:
        trained_model (dict): A dictionary containing:
            - "pipeline": A scikit-learn Pipeline with a TfidfVectorizer and a MultiOutputClassifier.
            - "le_priority": LabelEncoder for Priority.
            - "le_taxonomy": LabelEncoder for Taxonomy.
        description (str): The raw text of the alert's description.
        
    Returns:
        dict: A dictionary with keys "Priority" and "Taxonomy" holding the predicted labels.
    """
    pipeline = trained_model["pipeline"]
    le_priority = trained_model["le_priority"]
    le_taxonomy = trained_model["le_taxonomy"]
    
    # The pipeline expects an iterable of texts.
    prediction = pipeline.predict([description])  # prediction shape: (1, 2)
    priority_numeric, taxonomy_numeric = prediction[0]
    
    # Decode numeric labels to original string labels.
    priority_label = le_priority.inverse_transform([priority_numeric])[0]
    taxonomy_label = le_taxonomy.inverse_transform([taxonomy_numeric])[0]
    
    return {"Priority": priority_label, "Taxonomy": taxonomy_label}
