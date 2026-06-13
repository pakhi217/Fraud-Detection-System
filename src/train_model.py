"""
train_model.py
--------------
Trains Logistic Regression, Random Forest, and XGBoost on the fraud dataset.
Automatically selects the best model by F1-score and saves it to disk.

Run:
    python src/train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

# Allow imports from src/ when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import preprocess
from feature_engineering import prepare_data

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR        = "models"
BEST_MODEL_PATH   = os.path.join(MODELS_DIR, "fraud_model.pkl")
METRICS_CSV_PATH  = os.path.join(MODELS_DIR, "model_metrics.csv")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
RANDOM_STATE      = 42


def get_models() -> dict:
    """
    Return a dictionary of model name → unfitted estimator.
    Hyperparameters are chosen for a good speed/accuracy trade-off.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",   # further penalises misclassified minority
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,                 # use all CPU cores
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    models: dict,
    X_train, y_train,
    X_test,  y_test,
) -> pd.DataFrame:
    """
    Train each model and collect evaluation metrics.

    Metrics used (not plain accuracy — see WHY below):
    --------------------------------------------------
    Precision : Of all predicted frauds, how many were actually fraud?
    Recall    : Of all actual frauds, how many did we catch?
    F1-score  : Harmonic mean of Precision & Recall — primary selection metric.
    ROC-AUC   : Area under the ROC curve — overall ranking ability.

    WHY NOT ACCURACY?
    Plain accuracy is misleading when classes are imbalanced.
    e.g. always predicting "Normal" gives 99.83% accuracy but catches 0 fraud.
    """
    results = []

    for name, model in models.items():
        print(f"\n[TRAIN] Fitting {name} ...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]   # fraud probability

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred,    zero_division=0)
        f1        = f1_score(y_test, y_pred,        zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_proba)

        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")

        results.append({
            "Model":     name,
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "F1":        round(f1,        4),
            "ROC-AUC":   round(roc_auc,   4),
            "_model_obj": model,          # kept in memory, not saved to CSV
        })

    return pd.DataFrame(results)


def select_best_model(metrics_df: pd.DataFrame) -> tuple:
    """Select the model with the highest F1-score."""
    best_idx   = metrics_df["F1"].idxmax()
    best_row   = metrics_df.loc[best_idx]
    best_model = best_row["_model_obj"]
    best_name  = best_row["Model"]
    print(f"\n[INFO] ✅ Best model: {best_name} (F1={best_row['F1']:.4f})")
    return best_name, best_model


def save_artifacts(best_model, feature_names: list, metrics_df: pd.DataFrame) -> None:
    """Persist the best model, feature names, and metrics CSV."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"[INFO] Model saved → {BEST_MODEL_PATH}")

    # Save feature names (needed by the API / Streamlit app)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    print(f"[INFO] Feature names saved → {FEATURE_NAMES_PATH}")

    # Save metrics (drop the Python object column before writing CSV)
    metrics_df.drop(columns=["_model_obj"]).to_csv(METRICS_CSV_PATH, index=False)
    print(f"[INFO] Metrics saved → {METRICS_CSV_PATH}")


def main():
    print("=" * 60)
    print("  Fraud Detection — Model Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess raw data (creates processed CSV + scaler)
    print("\n[STEP 1] Preprocessing data ...")
    df = preprocess()

    # Step 2: Feature engineering (split + SMOTE)
    print("\n[STEP 2] Feature engineering ...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    feature_names = list(
        df.drop(columns=["Class"]).columns
    )

    # Step 3: Train all models
    print("\n[STEP 3] Training models ...")
    models     = get_models()
    metrics_df = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    # Step 4: Print comparison table
    print("\n[STEP 4] Model comparison:")
    print(metrics_df.drop(columns=["_model_obj"]).to_string(index=False))

    # Step 5: Select best and save
    print("\n[STEP 5] Selecting and saving best model ...")
    best_name, best_model = select_best_model(metrics_df)
    save_artifacts(best_model, feature_names, metrics_df)

    print("\n✅ Training complete! Run the app with:")
    print("   streamlit run app/streamlit_app.py")
    print("   uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
