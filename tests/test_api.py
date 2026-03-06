# 📁 tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "deployment"))

# Import app
from simple_api import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns 200"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_prediction_endpoint():
    """Test prediction endpoint works"""
    test_customer = {
        "tenure": 12,
        "monthly_charges": 75.5,
        "contract_encoded": 2,
        "payment_risk": 3,
        "total_services": 3
    }
    
    response = client.post("/predict", json=test_customer)
    assert response.status_code == 200
    data = response.json()
    assert "success" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])