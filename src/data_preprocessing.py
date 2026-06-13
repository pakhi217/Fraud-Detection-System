"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and preparing the raw credit card fraud dataset.
Follows PEP8 style with full comments for beginner-friendliness.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join("data", "processed")
SCALER_PATH   = os.path.join("models", "scaler.pkl")


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download creditcard.csv from Kaggle and place it in data/raw/.\n"
            "See data/raw/README.md for instructions."
        )
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Report any missing values in the DataFrame."""
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("[INFO] No missing values found ✓")
    else:
        print("[WARN] Missing values detected:")
        print(missing[missing > 0])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and report count removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicate rows")
    else:
        print("[INFO] No duplicate rows found ✓")
    return df


def check_dtypes(df: pd.DataFrame) -> None:
    """Print a summary of column data types."""
    print("\n[INFO] Column data types:")
    print(df.dtypes.value_counts())


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Detect outliers in a column using the IQR method.
    Returns a dict with Q1, Q3, IQR, lower bound, upper bound, and outlier count.
    """
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df[column] < lower) | (df[column] > upper)).sum()
    return {
        "column": column,
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower_bound": lower, "upper_bound": upper,
        "outlier_count": int(outlier_count),
    }


def scale_features(df: pd.DataFrame, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale 'Time' and 'Amount' columns using StandardScaler.
    V1-V28 are already PCA-transformed (zero mean, unit variance).

    Args:
        df:  Input DataFrame (must contain 'Time' and 'Amount').
        fit: If True, fit a new scaler; if False, load existing scaler from disk.

    Returns:
        (scaled DataFrame, fitted scaler)
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
        # Save scaler so the API can use the same transformation
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to {SCALER_PATH}")
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at '{SCALER_PATH}'. Run train_model.py first."
            )
        scaler = joblib.load(SCALER_PATH)
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
        print("[INFO] Loaded existing scaler and transformed features")

    return df, scaler


def preprocess(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Check missing values
      3. Remove duplicates
      4. Check data types
      5. Scale Time & Amount
    Returns the cleaned, scaled DataFrame.
    """
    df = load_data(path)
    check_missing_values(df)
    df = remove_duplicates(df)
    check_dtypes(df)

    # Outlier info for Amount (informational only — we keep all rows)
    outlier_info = detect_outliers_iqr(df, "Amount")
    print(f"[INFO] Outliers in 'Amount': {outlier_info['outlier_count']:,} "
          f"(bounds: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}])")

    df, _ = scale_features(df, fit=True)

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "creditcard_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Processed data saved to {out_path}")

    return df


if __name__ == "__main__":
    preprocess()
