"""
Unit tests for the API
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data

def test_predict_single():
    """Test single prediction endpoint"""
    response = client.post(
        "/predict",
        json={"text": "This course is amazing!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 0 <= data["confidence"] <= 1

def test_predict_batch():
    """Test batch prediction endpoint"""
    response = client.post(
        "/predict/batch",
        json={
            "reviews": [
                "Great course!",
                "Terrible experience",
                "It was okay"
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 3
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 3

def test_predict_empty_text():
    """Test prediction with empty text"""
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422  # Validation error

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/models/info")
    assert response.status_code in [200, 503]  # 503 if models not loaded
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "sentiment_classes" in data
