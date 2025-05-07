# loan_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict, Optional
import logging

# Initialize FastAPI
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status using machine learning",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
MODEL_PATH = "models/random_forest.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")


# Pydantic models for request/response validation
class LoanApplication(BaseModel):
    income_annum: float
    loan_amount: float
    cibil_score: float
    residential_assets_value: Optional[float] = 0
    commercial_assets_value: Optional[float] = 0
    luxury_assets_value: Optional[float] = 0
    bank_asset_value: Optional[float] = 0
    self_employed: Optional[int] = 0
    education: Optional[str] = "Graduate"
    loan_term: Optional[float] = 12.0


class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication]


class PredictionResult(BaseModel):
    application_id: Optional[int]
    status: str
    probability: float
    risk_level: Optional[str]


# Helper functions
def preprocess_input(data: Dict) -> pd.DataFrame:
    """Convert input data to model-compatible format"""
    df = pd.DataFrame([data])

    # Feature engineering
    df['debt_to_income'] = df['loan_amount'] / df['income_annum']
    df['total_assets'] = (df['residential_assets_value'] +
                          df['commercial_assets_value'] +
                          df['luxury_assets_value'] +
                          df['bank_asset_value'])

    # Convert categoricals
    df['education'] = df['education'].map({"Graduate": 1, "Not Graduate": 0})

    return df


def calculate_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability > 0.8:
        return "Low"
    elif probability > 0.5:
        return "Medium"
    else:
        return "High"


# API Endpoints
@app.post("/predict", response_model=PredictionResult)
async def predict_single(application: LoanApplication):
    """Make prediction for single loan application"""
    try:
        input_df = preprocess_input(application.dict())
        proba = model.predict_proba(input_df)[0][1]

        return {
            "status": "Approved" if proba >= 0.5 else "Rejected",
            "probability": float(proba),
            "risk_level": calculate_risk_level(proba)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=List[PredictionResult])
async def predict_batch(batch_request: BatchPredictionRequest):
    """Process batch of loan applications"""
    results = []
    for idx, app in enumerate(batch_request.applications):
        try:
            input_df = preprocess_input(app.dict())
            proba = model.predict_proba(input_df)[0][1]

            results.append({
                "application_id": idx,
                "status": "Approved" if proba >= 0.5 else "Rejected",
                "probability": float(proba),
                "risk_level": calculate_risk_level(proba)
            })
        except Exception as e:
            logger.error(f"Failed processing application {idx}: {str(e)}")
            results.append({
                "application_id": idx,
                "status": "Error",
                "probability": 0.0,
                "risk_level": "Unknown"
            })

    return results


@app.get("/model_info")
async def get_model_info():
    """Return model metadata"""
    return {
        "model_type": str(type(model)),
        "features_used": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
        "model_version": "1.0"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)