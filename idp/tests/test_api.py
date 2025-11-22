"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)


class TestAPI:
    """Test cases for API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint_valid_data(self):
        """Test prediction endpoint with valid data."""
        patient_data = {
            "Pregnancies": 1,
            "Glucose": 120.0,
            "BloodPressure": 72.0,
            "SkinThickness": 20.0,
            "Insulin": 79.0,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.47,
            "Age": 29
        }

        response = client.post("/predict", json=patient_data)

        # If model is not loaded, it should return 503
        # If model is loaded, it should return 200
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1

    def test_predict_endpoint_invalid_data(self):
        """Test prediction endpoint with invalid data."""
        patient_data = {
            "Pregnancies": 1,
            "Glucose": 600.0,  # Out of range
            "BloodPressure": 72.0
            # Missing other fields
        }

        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint."""
        patients = [
            {
                "Pregnancies": 1,
                "Glucose": 120.0,
                "BloodPressure": 72.0,
                "SkinThickness": 20.0,
                "Insulin": 79.0,
                "BMI": 32.0,
                "DiabetesPedigreeFunction": 0.47,
                "Age": 29
            },
            {
                "Pregnancies": 2,
                "Glucose": 130.0,
                "BloodPressure": 80.0,
                "SkinThickness": 25.0,
                "Insulin": 85.0,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.52,
                "Age": 35
            }
        ]

        response = client.post("/batch-predict", json=patients)

        # If model is not loaded, it should return 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/model-info")

        # If model is not loaded, it should return 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
