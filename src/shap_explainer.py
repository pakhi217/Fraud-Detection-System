"""
shap_explainer.py
-----------------
Uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

SHAP answers: "Why did the model label this transaction as Fraud?"
It assigns each feature a contribution score (positive = towards Fraud,
negative = towards Normal).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import get_model, get_feature_names

# ── Config ────────────────────────────────────────────────────────────────────
SCALER_PATH = os.path.join("models", "scaler.pkl")
PLOTS_DIR   = os.path.join("screenshots")


def _get_explainer(model):
    """
    Create the appropriate SHAP explainer for the given model type.

    - TreeExplainer  : fast, exact — for tree-based models (RF, XGBoost)
    - LinearExplainer: fast, exact — for linear models (Logistic Regression)
    - KernelExplainer: slow, model-agnostic fallback
    """
    model_name = type(model).__name__
    if model_name in ("RandomForestClassifier", "XGBClassifier",
                      "GradientBoostingClassifier", "DecisionTreeClassifier"):
        return shap.TreeExplainer(model)
    elif model_name in ("LogisticRegression", "LinearSVC", "Ridge"):
        return shap.LinearExplainer(model, feature_perturbation="interventional")
    else:
        # Generic fallback — slow but works for any model
        return shap.KernelExplainer(model.predict_proba, shap.sample(np.zeros((1, 30)), 50))


def explain_transaction(transaction_dict: dict) -> dict:
    """
    Generate SHAP values for a single transaction.

    Args:
        transaction_dict: Feature name → value mapping (raw, unscaled values).

    Returns:
        {
            "shap_values" : list of per-feature SHAP values (float),
            "feature_names": list of feature name strings,
            "base_value"  : float (expected model output),
            "top_positive": [(feature, shap_val), ...],  # top fraud drivers
            "top_negative": [(feature, shap_val), ...],  # top normal drivers
        }
    """
    model         = get_model()
    feature_names = get_feature_names()
    scaler        = joblib.load(SCALER_PATH)

    # Build DataFrame in correct feature order
    df = pd.DataFrame([transaction_dict])[feature_names]

    # Apply the same scaling used during training
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    explainer   = _get_explainer(model)
    shap_values = explainer.shap_values(df)

    # For binary classifiers, shap_values is a list [class0, class1]
    # We want class-1 (fraud) SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1][0]          # shape: (n_features,)
        base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") \
                   else explainer.expected_value
    else:
        sv       = shap_values[0]
        base_val = explainer.expected_value

    sv_list = sv.tolist()

    # Sort features by |SHAP value| descending
    sorted_pairs = sorted(zip(feature_names, sv_list), key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(f, v) for f, v in sorted_pairs if v > 0][:5]
    top_negative = [(f, v) for f, v in sorted_pairs if v < 0][:5]

    return {
        "shap_values":   sv_list,
        "feature_names": feature_names,
        "base_value":    float(base_val),
        "top_positive":  top_positive,
        "top_negative":  top_negative,
    }


def plot_shap_bar(explanation: dict, save_path: str | None = None) -> plt.Figure:
    """
    Create a horizontal bar chart of the top 10 most influential features.
    Red bars push towards Fraud; Blue bars push towards Normal.

    Returns the matplotlib Figure (for embedding in Streamlit).
    """
    feature_names = explanation["feature_names"]
    shap_values   = explanation["shap_values"]

    # Take top 10 by absolute SHAP value
    pairs  = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:10]
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("SHAP Value  (← Normal | Fraud →)", fontsize=11)
    ax.set_title("Feature Contributions to Prediction", fontsize=13, fontweight="bold")

    # Annotate each bar with its value
    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width()
        ax.text(
            x_pos + (0.002 if x_pos >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if x_pos >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] SHAP plot saved → {save_path}")

    return fig


if __name__ == "__main__":
    # Demo using a dummy all-zero transaction
    feature_names = get_feature_names()
    dummy = {f: 0.0 for f in feature_names}
    dummy["Amount"] = 200.0
    dummy["Time"]   = 7200.0

    explanation = explain_transaction(dummy)
    print("\nTop features PUSHING TOWARDS FRAUD:")
    for feat, val in explanation["top_positive"]:
        print(f"  {feat:10s}  {val:+.4f}")
    print("\nTop features PUSHING TOWARDS NORMAL:")
    for feat, val in explanation["top_negative"]:
        print(f"  {feat:10s}  {val:+.4f}")

    plot_shap_bar(explanation, save_path=os.path.join(PLOTS_DIR, "shap_explanation.png"))
