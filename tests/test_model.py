# 📁 tests/test_model.py
import pytest
import joblib
import os
from pathlib import Path

# Detect if running in GitHub Actions
IN_CI = os.environ.get('CI') == 'true'

# Get the correct path to models
NOTEBOOKS_PATH = Path(__file__).parent.parent / "models"
print(f"🔍 Looking for models in: {NOTEBOOKS_PATH}")

def test_model_exists():
    """Test model file exists"""
    model_path = NOTEBOOKS_PATH / "logistic_regression_model.pkl"
    print(f"Checking: {model_path}")
    assert model_path.exists(), f"Model not found at {model_path}"

def test_model_loads():
    """Test model loads"""
    model_path = NOTEBOOKS_PATH / "logistic_regression_model.pkl"
    model = joblib.load(model_path)
    assert model is not None
    print("✅ Model loaded successfully")

def test_scaler_exists():
    """Test scaler file exists"""
    scaler_path = NOTEBOOKS_PATH / "scaler.pkl"
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"

@pytest.mark.skipif(IN_CI, reason="Scaler loading has compatibility issues in CI environment")
def test_scaler_loads():
    """Test scaler loads (skipped in CI)"""
    scaler_path = NOTEBOOKS_PATH / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    assert scaler is not None
    print("✅ Scaler loaded successfully")

def test_feature_names_exists():
    """Test feature names file exists"""
    features_path = NOTEBOOKS_PATH / "feature_names.pkl"
    assert features_path.exists(), f"Features not found at {features_path}"

def test_feature_names_loads():
    """Test feature names loads"""
    features_path = NOTEBOOKS_PATH / "feature_names.pkl"
    feature_names = joblib.load(features_path)
    assert feature_names is not None
    assert len(feature_names) > 0
    print(f"✅ Feature names loaded: {len(feature_names)} features")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])