# 📁 deployment/simple_api.py - FIXED VERSION
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import uvicorn
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability",
    version="1.0.0"
)

# FIXED: Don't exit! Just set model to None and continue
print("📦 Loading models from notebooks folder...")

notebooks_path = Path("../notebooks")
model = None
scaler = None
feature_names = None

# Check if files exist - but DON'T exit!
model_file = notebooks_path / "logistic_regression_model.pkl"
scaler_file = notebooks_path / "scaler.pkl"
features_file = notebooks_path / "feature_names.pkl"

if not model_file.exists():
    print(f"⚠️ Model not found at: {model_file.absolute()}")
    print("⚠️ API will run in DEGRADED mode (predictions won't work)")
else:
    try:
        # Load the files
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        feature_names = joblib.load(features_file)
        print(f"✅ Loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"📋 Number of features: {len(feature_names)}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️ API will run in DEGRADED mode")

# Define input model
class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract_encoded: int
    payment_risk: int
    total_services: int

@app.get("/")
def root():
    return {
        "message": "🔮 Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model": "loaded" if model else "degraded"
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    """Predict churn probability for a single customer"""
    if model is None or scaler is None or feature_names is None:
        return {
            "success": False,
            "error": "Model not loaded. API running in degraded mode.",
            "churn_probability": 0.5,
            "risk_level": "UNKNOWN"
        }
    
    try:
        # Create dataframe with ALL features (fill missing with 0)
        data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the values we have
        if 'tenure' in feature_names:
            data['tenure'] = customer.tenure
        if 'MonthlyCharges' in feature_names:
            data['MonthlyCharges'] = customer.monthly_charges
        if 'Contract_encoded' in feature_names:
            data['Contract_encoded'] = customer.contract_encoded
        if 'PaymentRisk' in feature_names:
            data['PaymentRisk'] = customer.payment_risk
        if 'TotalServices' in feature_names:
            data['TotalServices'] = customer.total_services
        
        # Scale the features
        data_scaled = scaler.transform(data)
        
        # Make prediction
        probability = model.predict_proba(data_scaled)[0][1]
        
        # Determine risk level
        if probability > 0.7:
            risk = "HIGH"
            action = "🚨 Immediate retention offer needed"
        elif probability > 0.3:
            risk = "MEDIUM"
            action = "📧 Send engagement email"
        else:
            risk = "LOW"
            action = "✅ Regular maintenance"
        
        return {
            "success": True,
            "churn_probability": round(float(probability), 4),
            "churn_prediction": bool(probability > 0.5),
            "risk_level": risk,
            "recommended_action": action,
            "model_used": "logistic_regression"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Remove the __main__ block that calls uvicorn.run() when testing
# This prevents pytest from trying to start the server
if __name__ == "__main__":
    # Only run server when script is executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)