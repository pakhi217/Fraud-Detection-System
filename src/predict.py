"""
predict.py
----------
Prediction utilities used by both the Streamlit app and FastAPI.
Loads the saved model + scaler and exposes a simple predict() function.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH         = os.path.join("models", "fraud_model.pkl")
SCALER_PATH        = os.path.join("models", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("models", "feature_names.json")

# Cache loaded artifacts in module-level variables to avoid repeated disk reads
_model         = None
_scaler        = None
_feature_names = None


def _load_artifacts():
    """Load model, scaler, and feature names from disk (lazy, cached)."""
    global _model, _scaler, _feature_names

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Please run:  python src/train_model.py"
            )
        _model         = joblib.load(MODEL_PATH)
        _scaler        = joblib.load(SCALER_PATH)
        with open(FEATURE_NAMES_PATH) as f:
            _feature_names = json.load(f)


def predict(transaction: dict) -> dict:
    """
    Predict whether a transaction is fraudulent.

    Args:
        transaction: A dict mapping feature names to their values.
                     Must contain 'Time', 'Amount', and V1–V28.

    Returns:
        {
            "prediction"  : "Fraud" | "Normal",
            "risk_score"  : float (0.0 – 1.0),
            "label"       : 1 | 0,
        }
    """
    _load_artifacts()

    # Build a single-row DataFrame in the expected column order
    df = pd.DataFrame([transaction])[_feature_names]

    # Scale 'Time' and 'Amount' using the saved scaler
    df[["Time", "Amount"]] = _scaler.transform(df[["Time", "Amount"]])

    # Get class probabilities: index 1 = P(fraud)
    risk_score = float(_model.predict_proba(df)[0][1])
    label      = int(_model.predict(df)[0])

    return {
        "prediction": "Fraud" if label == 1 else "Normal",
        "risk_score": round(risk_score, 4),
        "label":      label,
    }


def predict_from_array(X: np.ndarray) -> np.ndarray:
    """
    Batch predict from a NumPy array (already scaled).
    Used internally by shap_explainer.py.
    """
    _load_artifacts()
    return _model.predict_proba(X)[:, 1]


def get_model():
    """Return the loaded model object (for SHAP explainer)."""
    _load_artifacts()
    return _model


def get_feature_names() -> list:
    """Return the ordered list of feature names."""
    _load_artifacts()
    return _feature_names


if __name__ == "__main__":
    # Quick smoke-test with a dummy transaction
    dummy = {col: 0.0 for col in get_feature_names()}
    dummy["Amount"] = 150.0
    dummy["Time"]   = 3600.0
    result = predict(dummy)
    print("Dummy prediction:", result)
