"""
feature_engineering.py
-----------------------
Prepares features for model training:
  - Separates features (X) and target (y)
  - Applies train/test split
  - Applies SMOTE to handle class imbalance
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join("data", "processed", "creditcard_processed.csv")
TARGET_COLUMN       = "Class"
TEST_SIZE           = 0.2       # 20% held out for testing
RANDOM_STATE        = 42        # reproducibility seed
SMOTE_RANDOM_STATE  = 42


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed CSV produced by data_preprocessing.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at '{path}'.\n"
            "Run data_preprocessing.py first, or call preprocess() from that module."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded processed data: {len(df):,} rows")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into:
      X — all feature columns (everything except 'Class')
      y — target column ('Class': 0=Normal, 1=Fraud)
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"[INFO] Features: {X.shape[1]} columns | Target distribution:")
    print(y.value_counts().rename({0: "Normal", 1: "Fraud"}).to_string())
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Stratified train/test split.
    Stratify=y ensures both splits maintain the original fraud ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,        # preserve class ratio in both splits
    )
    print(f"[INFO] Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = SMOTE_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to the training set.

    WHY SMOTE?
    ----------
    Only ~0.17% of transactions are fraudulent. A naive model that always predicts
    "Normal" would be 99.83% accurate — yet completely useless for fraud detection.

    SMOTE creates *synthetic* fraud samples by interpolating between real fraud
    examples in feature space, balancing the training set without simply
    duplicating existing samples.

    NOTE: SMOTE is applied ONLY to training data to prevent data leakage.
    """
    print(f"\n[INFO] Before SMOTE — Fraud: {y_train.sum():,} | Normal: {(y_train==0).sum():,}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    fraud_count  = int(np.sum(y_resampled == 1))
    normal_count = int(np.sum(y_resampled == 0))
    print(f"[INFO] After  SMOTE — Fraud: {fraud_count:,} | Normal: {normal_count:,}")
    print(f"[INFO] Total training samples after SMOTE: {len(X_resampled):,}")

    return X_resampled, y_resampled


def prepare_data(df: pd.DataFrame | None = None) -> tuple:
    """
    Full feature-engineering pipeline.
    If df is None, loads the processed CSV from disk.

    Returns:
        X_train_smote, X_test, y_train_smote, y_test
    """
    if df is None:
        df = load_processed_data()

    X, y                              = split_features_target(df)
    X_train, X_test, y_train, y_test  = train_test_split_data(X, y)
    X_train_s, y_train_s              = apply_smote(X_train, y_train)

    return X_train_s, X_test, y_train_s, y_test


if __name__ == "__main__":
    prepare_data()
