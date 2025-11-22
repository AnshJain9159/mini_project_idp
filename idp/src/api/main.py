"""FastAPI application for diabetes prediction."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import setup_logger, get_logger
from utils.config_loader import get_config

# Setup logger
setup_logger('diabetes_api', log_level='INFO')
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="AI-powered diabetes risk prediction with interpretability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PatientData(BaseModel):
    """Patient medical data for prediction."""
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=500, description="Glucose level (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    Insulin: float = Field(..., ge=0, le=900, description="Insulin level (mu U/ml)")
    BMI: float = Field(..., ge=0, le=70, description="Body Mass Index (kg/m²)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5, description="Diabetes pedigree function")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 1,
                "Glucose": 120,
                "BloodPressure": 72,
                "SkinThickness": 20,
                "Insulin": 79,
                "BMI": 32.0,
                "DiabetesPedigreeFunction": 0.47,
                "Age": 29
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int = Field(..., description="Prediction (0: No diabetes, 1: Diabetes)")
    probability: float = Field(..., description="Probability of diabetes")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High/Critical)")
    confidence: float = Field(..., description="Model confidence")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool


# Global variables for model and preprocessor
model = None
preprocessor = None


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup."""
    global model, preprocessor

    try:
        model_path = os.path.join('models', 'xgboost_diabetes_model.joblib')
        scaler_path = os.path.join('models', 'scaler.joblib')

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}")

        if os.path.exists(scaler_path):
            from data.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.load_transformers('models')
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Make diabetes prediction for a patient.

    Args:
        patient_data: Patient medical data

    Returns:
        Prediction with probability and risk level
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained and saved."
        )

    try:
        # Convert to DataFrame
        data_dict = patient_data.dict()
        df = pd.DataFrame([data_dict])

        # Preprocess if preprocessor is available
        if preprocessor is not None:
            # Apply preprocessing (without fitting)
            df_processed, _ = preprocessor.preprocess_pipeline(
                df, None,
                fit=False,
                handle_imbalance=False,
                remove_outliers_flag=False
            )
        else:
            # Use scaler directly if available
            scaler_path = os.path.join('models', 'scaler.joblib')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                df_processed = pd.DataFrame(
                    scaler.transform(df),
                    columns=df.columns
                )
            else:
                df_processed = df

        # Make prediction
        prediction = int(model.predict(df_processed)[0])
        probability = float(model.predict_proba(df_processed)[0][1])

        # Determine risk level
        risk_thresholds = get_config('risk_stratification.thresholds', {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        })

        if probability < risk_thresholds.get('low', 0.3):
            risk_level = "Low"
        elif probability < risk_thresholds.get('medium', 0.5):
            risk_level = "Medium"
        elif probability < risk_thresholds.get('high', 0.7):
            risk_level = "High"
        else:
            risk_level = "Critical"

        # Calculate confidence (distance from decision boundary at 0.5)
        confidence = abs(probability - 0.5) * 2

        logger.info(f"Prediction made: {prediction} (probability: {probability:.4f}, risk: {risk_level})")

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict")
async def batch_predict(patients: List[PatientData]):
    """
    Make predictions for multiple patients.

    Args:
        patients: List of patient medical data

    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        results = []
        for patient in patients:
            result = await predict(patient)
            results.append(result)

        logger.info(f"Batch prediction completed for {len(patients)} patients")
        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        metadata_path = os.path.join('models', 'training_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            return {
                "model_type": metadata.get('model_type', 'unknown'),
                "best_cv_score": metadata.get('best_cv_score'),
                "preprocessing_steps": metadata.get('preprocessing_steps', []),
                "calibration_method": metadata.get('calibration_method')
            }
        else:
            return {
                "model_type": "xgboost",
                "status": "Model loaded but metadata not available"
            }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
