# 🛡️ Fraud Detection System using Machine Learning

> An end-to-end ML pipeline that detects fraudulent credit card transactions in real time, explains predictions with SHAP, and exposes a REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Architecture](#-architecture)
4. [Features](#-features)
5. [Technologies Used](#-technologies-used)
6. [Project Structure](#-project-structure)
7. [Dataset](#-dataset)
8. [Installation](#-installation)
9. [How to Run](#-how-to-run)
10. [Model Performance](#-model-performance)
11. [API Reference](#-api-reference)
12. [Screenshots](#-screenshots)
13. [Future Improvements](#-future-improvements)
14. [Author](#-author)

---

## 🎯 Project Overview

This project builds a complete, production-ready fraud detection system that:

- **Predicts** whether a credit card transaction is fraudulent (binary classification)
- **Scores** each transaction with a fraud risk probability (0.0 – 1.0)
- **Explains** predictions at the feature level using SHAP values
- **Visualises** results through an interactive Streamlit dashboard
- **Exposes** a REST API endpoint via FastAPI for integration into real systems

---

## 🧩 Problem Statement

Credit card fraud causes billions in losses annually. The challenge:

- Fraud is **extremely rare** (~0.17% of transactions) → severe class imbalance
- Plain accuracy is **misleading** (a model predicting "Normal" for everything scores 99.83%)
- Real systems need **high recall** (catch most fraud) and **explainability** (why was this flagged?)

This system addresses all three challenges using SMOTE, F1-optimised model selection, and SHAP explanations.

---

## 🏗️ Architecture

```
Raw CSV
   │
   ▼
Data Preprocessing ──► StandardScaler ──► Processed CSV
   │
   ▼
Feature Engineering
   ├── Train/Test Split (80/20, stratified)
   └── SMOTE (balance training set)
          │
          ▼
    Model Training
    ├── Logistic Regression
    ├── Random Forest
    └── XGBoost  ◄── Best model (by F1-score)
          │
          ▼
   Saved Artifacts
   ├── fraud_model.pkl
   ├── scaler.pkl
   └── feature_names.json
          │
     ┌────┴────┐
     ▼         ▼
Streamlit    FastAPI
Dashboard    REST API
     │         │
     └────┬────┘
          ▼
     SHAP Explainer
   (Why was this Fraud?)
```

---

## ✨ Features

| Feature                  | Description                                              |
|--------------------------|----------------------------------------------------------|
| 🔍 Real-time Prediction   | Classify a transaction as Fraud or Normal instantly      |
| 📊 Risk Score             | Probability-based fraud risk (0.0 – 1.0)                |
| 🧠 SHAP Explanations      | Feature-level explanation for every prediction           |
| 📈 Risk Gauge             | Visual semicircular risk meter in the dashboard          |
| 📉 Evaluation Plots       | Confusion matrix, ROC curve, classification report       |
| 🌐 REST API               | FastAPI with Swagger UI at `/docs`                      |
| 📓 Notebooks              | Jupyter notebooks for EDA, preprocessing, and training  |
| ⚖️ Class Imbalance        | SMOTE + class weighting to handle 0.17% fraud rate      |

---

## 🛠️ Technologies Used

| Category          | Tools                                              |
|-------------------|----------------------------------------------------|
| Language          | Python 3.10+                                       |
| Data              | Pandas, NumPy                                      |
| Visualisation     | Matplotlib, Seaborn                                |
| ML                | Scikit-learn, XGBoost                              |
| Imbalance         | imbalanced-learn (SMOTE)                           |
| Explainability    | SHAP                                               |
| Web App           | Streamlit                                          |
| API               | FastAPI + Uvicorn                                  |
| Persistence       | Joblib / Pickle                                    |

---

## 📁 Project Structure

```
Fraud-Detection-System/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          ← Download from Kaggle (see README)
│   └── processed/
│       └── creditcard_processed.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb      ← Cleaning, scaling, SMOTE
│   └── 03_Model_Training.ipynb     ← Train, evaluate, compare models
│
├── src/
│   ├── data_preprocessing.py       ← Load, clean, scale data
│   ├── feature_engineering.py      ← Split + SMOTE
│   ├── train_model.py              ← Full training pipeline
│   ├── evaluate_model.py           ← Plots: confusion matrix, ROC
│   ├── predict.py                  ← Prediction utility (model + scaler)
│   └── shap_explainer.py           ← SHAP values + bar plot
│
├── models/
│   ├── fraud_model.pkl             ← Best trained model
│   ├── scaler.pkl                  ← Fitted StandardScaler
│   ├── feature_names.json          ← Ordered feature list
│   └── model_metrics.csv           ← All models' metrics
│
├── app/
│   ├── streamlit_app.py            ← Interactive dashboard
│   └── dashboard.py                ← Chart-generation helpers
│
├── api/
│   └── main.py                     ← FastAPI REST API
│
├── screenshots/                    ← Generated plots & app screenshots
├── sample_transaction.json         ← Example API request body
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 📦 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property        | Value                            |
|-----------------|----------------------------------|
| Total Rows      | 284,807 transactions             |
| Fraud Cases     | 492 (0.172%)                     |
| Normal Cases    | 284,315 (99.828%)                |
| Features        | Time, V1–V28 (PCA), Amount       |
| Target Column   | Class (0 = Normal, 1 = Fraud)    |
| File Size       | ~150 MB                          |

**Why is this dataset imbalanced?**
Real fraud is rare. A naive model predicting "Normal" always would achieve 99.83% accuracy — yet catch zero fraud cases. This is why we use **SMOTE** and evaluate with **Precision, Recall, and F1-score** instead of accuracy.

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Fraud-Detection-System.git
cd Fraud-Detection-System

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    → Place creditcard.csv in data/raw/
```

---

## ▶️ How to Run

### Step 1 — Train the Model

```bash
python src/train_model.py
```

This will:
- Preprocess the raw CSV
- Apply SMOTE
- Train Logistic Regression, Random Forest, and XGBoost
- Select the best model by F1-score
- Save `models/fraud_model.pkl`, `models/scaler.pkl`, `models/model_metrics.csv`

### Step 2 — Generate Evaluation Plots

```bash
python src/evaluate_model.py
```

Saves confusion matrix and ROC curve to `screenshots/`.

### Step 3 — Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open your browser at: **http://localhost:8501**

### Step 4 — Launch the FastAPI Server

```bash
uvicorn api.main:app --reload
```

- API: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

### Step 5 — Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

Or use the Swagger UI at `/docs` for interactive testing.

### Run Notebooks

```bash
jupyter notebook notebooks/
```

---

## 📊 Model Performance

> Metrics shown after training on SMOTE-resampled data, evaluated on the original test set.

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | ~0.06     | ~0.92  | ~0.11    | ~0.97   |
| Random Forest       | ~0.96     | ~0.80  | ~0.87    | ~0.97   |
| **XGBoost** ✅      | **~0.93** | **~0.84** | **~0.88** | **~0.98** |

*Exact values will appear in `models/model_metrics.csv` after training.*

**Why F1-score for model selection?**
F1 balances Precision (avoid false alarms) and Recall (catch real fraud). Neither alone is sufficient in fraud detection.

---

## 🌐 API Reference

### `POST /predict`

Classify a transaction as Fraud or Normal.

**Request Body:**
```json
{
  "Time": 3600,
  "Amount": 149.62,
  "V1": -1.36, "V2": -0.07, "V3": 2.54,
  "V4": 1.38,  "V5": -0.34, "V6": 0.46,
  "V7": 0.24,  "V8": 0.10,  "V9": 0.36,
  "V10": -0.08, "V11": -0.27, "V12": -0.17,
  "V13": 0.12,  "V14": -0.07, "V15": -0.20,
  "V16": -0.24, "V17": 0.02,  "V18": 0.26,
  "V19": 0.09,  "V20": 0.02,  "V21": 0.02,
  "V22": -0.01, "V23": -0.01, "V24": 0.01,
  "V25": 0.002, "V26": -0.002,"V27": -0.002,
  "V28": 0.0003
}
```

**Response:**
```json
{
  "prediction": "Normal",
  "risk_score": 0.0432,
  "label": 0,
  "risk_level": "LOW"
}
```

### `GET /metrics`
Returns model comparison metrics as JSON.

### `GET /features`
Returns the list of expected feature names.

---



## 🔮 Future Improvements

- [ ] **Real-time streaming** — Integrate with Kafka for live transaction feeds
- [ ] **Threshold tuning** — Optimise the decision threshold for business requirements
- [ ] **Model retraining pipeline** — Automated retraining when fraud patterns shift
- [ ] **Docker deployment** — Containerise the API and dashboard
- [ ] **User authentication** — Secure the API with JWT tokens
- [ ] **Database integration** — Store predictions and audit logs in PostgreSQL
- [ ] **A/B testing** — Compare model versions in production
- [ ] **Alert system** — Email/Slack notifications for high-risk transactions

---

## 👤 Author

**PAKHI SAXENA**


## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data preprocessing, class imbalance handling, multi-model training, explainability, and deployment.*
