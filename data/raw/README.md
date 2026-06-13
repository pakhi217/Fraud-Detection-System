# Dataset Instructions

## Download the Credit Card Fraud Detection Dataset

This project uses the **Kaggle Credit Card Fraud Detection** dataset.

### Steps to Download:

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in to your Kaggle account (free registration required)
3. Click the **Download** button
4. Extract the ZIP file
5. Place `creditcard.csv` in this folder: `data/raw/creditcard.csv`

---

## Dataset Details

| Property        | Value                          |
|-----------------|-------------------------------|
| Total Rows      | 284,807 transactions           |
| Total Columns   | 31 features + 1 target         |
| Fraud Cases     | 492 (0.172% of all transactions)|
| Normal Cases    | 284,315                        |
| File Size       | ~150 MB                        |

## Features

- **Time**: Seconds elapsed between this transaction and the first transaction
- **V1–V28**: PCA-transformed anonymized features (to protect user privacy)
- **Amount**: Transaction amount in euros
- **Class**: Target variable → `0` = Normal, `1` = Fraud

## Why Is This Dataset Imbalanced?

Real-world fraud is rare. Only ~0.17% of transactions are fraudulent.
This makes standard accuracy a misleading metric — a model that predicts
"Normal" for every transaction would achieve 99.83% accuracy but catch
zero fraud cases!

This is why we use:
- **SMOTE** to oversample the minority (fraud) class
- **Precision, Recall, F1-score** as evaluation metrics
- **ROC-AUC** to measure overall discrimination ability
