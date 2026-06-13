"""
main.py  —  FastAPI REST API for Fraud Detection
-------------------------------------------------
Run:
    uvicorn api.main:app --reload

Endpoints:
    GET  /           → health check
    POST /predict    → fraud prediction + risk score
    GET  /metrics    → model performance metrics
    GET  /features   → list of expected feature names
"""

import os
import sys
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict, get_feature_names

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using an XGBoost model trained "
        "on the Kaggle Credit Card Fraud dataset. Returns a fraud probability "
        "score and prediction label."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Input schema: all 30 features of a credit card transaction."""
    Time:   float = Field(..., description="Seconds since first transaction in dataset")
    Amount: float = Field(..., description="Transaction amount in euros", ge=0)

    # PCA-anonymised features V1–V28
    V1:  float = 0.0; V2:  float = 0.0; V3:  float = 0.0; V4:  float = 0.0
    V5:  float = 0.0; V6:  float = 0.0; V7:  float = 0.0; V8:  float = 0.0
    V9:  float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 3600,
                "Amount": 149.62,
                "V1": -1.3598071336738, "V2": -0.0727811733098497,
                "V3": 2.53634673796914,  "V4": 1.37815522427443,
                "V5": -0.338320769942518,"V6": 0.462387777762292,
                "V7": 0.239598554061257, "V8": 0.0986979012610507,
                "V9": 0.363786969611213, "V10":-0.0827840797307893,
                "V11": -0.1, "V12": -0.1, "V13": 0.1,
                "V14": -0.1, "V15": 0.0, "V16": 0.0, "V17": 0.0,
                "V18": 0.0, "V19": 0.0, "V20": 0.0, "V21": 0.0,
                "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
                "V26": 0.0, "V27": 0.0, "V28": 0.0,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""
    prediction: str   = Field(..., description="'Fraud' or 'Normal'")
    risk_score: float = Field(..., description="Probability of fraud (0.0 – 1.0)")
    label:      int   = Field(..., description="1 = Fraud, 0 = Normal")
    risk_level: str   = Field(..., description="HIGH / MEDIUM / LOW")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Fraud Detection API is running 🛡️"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_endpoint(transaction: TransactionRequest):
    """
    Predict whether a credit card transaction is fraudulent.

    - **Time**: seconds elapsed since the first transaction in the dataset
    - **Amount**: transaction value in euros
    - **V1–V28**: PCA-anonymised features

    Returns a fraud prediction, probability score, and risk level.
    """
    try:
        # Convert Pydantic model to plain dict for the predict function
        transaction_dict = transaction.model_dump()
        result = predict(transaction_dict)

        # Classify risk level
        score = result["risk_score"]
        if score > 0.70:
            risk_level = "HIGH"
        elif score > 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            label=result["label"],
            risk_level=risk_level,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run 'python src/train_model.py' first. ({e})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Model Info"])
def get_metrics():
    """Return the saved model comparison metrics as JSON."""
    metrics_path = os.path.join("models", "model_metrics.csv")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run 'python src/train_model.py' first."
        )
    import pandas as pd
    df = pd.read_csv(metrics_path)
    return {"metrics": df.to_dict(orient="records")}


@app.get("/features", tags=["Model Info"])
def list_features():
    """Return the list of feature names expected by the model."""
    try:
        features = get_feature_names()
        return {"features": features, "count": len(features)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
