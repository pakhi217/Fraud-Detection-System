"""
evaluate_model.py
-----------------
Generates evaluation plots for the best trained model:
  - Confusion matrix
  - Classification report
  - ROC curve with AUC score

Run:
    python src/evaluate_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script usage
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess
from feature_engineering import prepare_data

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH         = os.path.join("models", "fraud_model.pkl")
FEATURE_NAMES_PATH = os.path.join("models", "feature_names.json")
PLOTS_DIR          = os.path.join("screenshots")


def load_model_and_features() -> tuple:
    """Load the saved model and feature name list."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    print(f"[INFO] Loaded model: {type(model).__name__}")
    return model, feature_names


def plot_confusion_matrix(y_test, y_pred, save_path: str) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Normal", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_roc_curve(y_test, y_proba, save_path: str) -> None:
    """Plot and save the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc      = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve",            fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved → {save_path}")


def evaluate():
    """Full evaluation pipeline."""
    print("=" * 60)
    print("  Fraud Detection — Model Evaluation")
    print("=" * 60)

    # Load data
    print("\n[STEP 1] Preprocessing data ...")
    df = preprocess()

    print("\n[STEP 2] Preparing features ...")
    _, X_test, _, y_test = prepare_data(df)

    # Load model
    model, feature_names = load_model_and_features()

    # Align columns
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Predictions
    y_pred  = model.predict(X_test_df)
    y_proba = model.predict_proba(X_test_df)[:, 1]

    # Classification report
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # Save plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba,
                   os.path.join(PLOTS_DIR, "roc_curve.png"))

    print("\n✅ Evaluation complete. Plots saved to screenshots/")


if __name__ == "__main__":
    evaluate()
