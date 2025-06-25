import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.optimize import curve_fit
from datetime import datetime

# Configs
INPUT_FILE = "data/dados.csv"  # Change this if your file has a different name
GRAPH_OUTPUT = "data/learning_curve.png"
DATE_COLUMN = "Date Created"
SEVERITY_MAP = {"LOW": "P3", "MEDIUM": "P2", "HIGH": "P1"}

# Helper: Logarithmic regression function
def log_func(x, a, b):
    return a * np.log(x) + b

def preprocess_dataframe(df):
    # Normalize column names: strip whitespace and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Print column names for debugging
    print("🧩 Available columns:", df.columns.tolist())

    # Rename relevant columns
    column_map = {
        "severity": "true_priority",
        "suggested priority": "pred_priority",
        "suggested taxonomy": "pred_taxonomy",
        "jira_taxonomy": "true_taxonomy",
        "date created": "created_at"
    }
    df = df.rename(columns=column_map)

    # ✅ Ensure all renamed columns exist now
    expected_columns = ["true_priority", "pred_priority", "pred_taxonomy", "true_taxonomy", "created_at"]
    for col in expected_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found after renaming. Check the Excel headers.")

    # Convert Severity to P1/P2/P3
    df["true_priority"] = df["true_priority"].str.upper().map(SEVERITY_MAP)

    df["pred_priority"] = df["pred_priority"].str.upper().map(SEVERITY_MAP)

    # Parse datetime
    df["created_at"] = pd.to_datetime(df["created_at"], format="%m/%d/%Y %H:%M:%S", errors="coerce")

    # Check failed rows (optional debug)
    # bad_dates = df[df["created_at"].isna()]
    # if not bad_dates.empty:
    #     print(f"⚠️ {len(bad_dates)} timestamps could not be parsed and will be excluded.")
    #     print("Example of bad raw strings:")
    #     print(bad_dates["created_at_raw"].head())

    # Drop them now
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at")

    first_time = df["created_at"].min()
    df["day_bin"] = ((df["created_at"] - first_time).dt.total_seconds() // (24 * 3600)).astype(int)

    print(f"\n📅 First prediction: {first_time}")
    print(f"📅 Last prediction: {df['created_at'].max()}")


    return df

def compute_statistics(df):
    print("\n=== PRIORITY METRICS ===")
    print("Accuracy:", accuracy_score(df["true_priority"], df["pred_priority"]))
    print("\nClassification Report (Priority):\n", classification_report(df["true_priority"], df["pred_priority"]))
    print("Confusion Matrix (Priority):\n", confusion_matrix(df["true_priority"], df["pred_priority"]))

    print("\n=== TAXONOMY METRICS ===")
    print("Accuracy:", accuracy_score(df["true_taxonomy"], df["pred_taxonomy"]))
    print("\nClassification Report (Taxonomy):\n", classification_report(df["true_taxonomy"], df["pred_taxonomy"]))
    print("Confusion Matrix (Taxonomy):\n", confusion_matrix(df["true_taxonomy"], df["pred_taxonomy"]))

def plot_learning_curve(df):
    df["correct"] = (df["true_priority"] == df["pred_priority"]) & (df["true_taxonomy"] == df["pred_taxonomy"])
    df["day_bin"] = df["day_bin"].astype(int)

    grouped = df.groupby("day_bin").agg({
        "correct": "sum",
        "pred_priority": "count"
    }).rename(columns={"correct": "correct_predictions", "pred_priority": "total_predictions"})

    grouped["day"] = grouped.index + 1
    grouped["accuracy"] = grouped["correct_predictions"] / grouped["total_predictions"]

    # ========== First Graph: Correct Predictions Per Day ==========
    x = grouped["day"].values
    y = grouped["correct_predictions"].values

    try:
        params, _ = curve_fit(log_func, x, y)
        y_fit = log_func(x, *params)
    except Exception as e:
        print("⚠️ Could not fit log curve:", e)
        y_fit = None

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=grouped["day"], y=grouped["correct_predictions"], label="Correct Predictions")
    if y_fit is not None:
        plt.plot(x, y_fit, color="red", label="Logarithmic Regression")
    plt.xlabel("Day since first alert")
    plt.ylabel("Correct Predictions")
    plt.title("Learning Curve: Correct Predictions per Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/learning_curve_corrects.png")
    print(f"📈 Correct predictions curve saved as 'learning_curve_corrects.png'")

    # ========== Second Graph: Accuracy Per Day ==========
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=grouped["day"], y=grouped["accuracy"], marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Day since first alert")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy per Day (Correct / Total)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/learning_curve_accuracy.png")
    print(f"📈 Accuracy per day graph saved as 'learning_curve_accuracy.png'")

    # Also print stats for CSV export if needed
    print("\n📊 Daily Statistics (Correct, Total, Accuracy):")
    print(grouped[["correct_predictions", "total_predictions", "accuracy"]])


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File '{INPUT_FILE}' not found.")
        return

    # df_raw = pd.read_excel(INPUT_FILE, header=None)
    # print(df_raw.head(10).to_string())

    df = pd.read_csv(INPUT_FILE, header=4, dtype={"Date Created": str})

    df = preprocess_dataframe(df)
    compute_statistics(df)
    plot_learning_curve(df)

if __name__ == "__main__":
    main()
