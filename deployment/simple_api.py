# 📁 CHURN_NEW_2/deployment/simple_api.py
# FIXED - NO CONFLICTS!

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

# FIXED: Load files from notebooks folder (one level up)
print("📦 Loading models from notebooks folder...")

notebooks_path = Path("../notebooks")

# Check if files exist
model_file = notebooks_path / "logistic_regression_model.pkl"
scaler_file = notebooks_path / "scaler.pkl"
features_file = notebooks_path / "feature_names.pkl"

if not model_file.exists():
    print(f"❌ Model not found at: {model_file.absolute()}")
    print("📁 Files in notebooks folder:")
    for f in notebooks_path.glob("*.pkl"):
        print(f"   - {f.name}")
    exit(1)

# Load the files
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
feature_names = joblib.load(features_file)

print(f"✅ Loaded successfully!")
print(f"📊 Model type: {type(model).__name__}")
print(f"📋 Number of features: {len(feature_names)}")
print(f"🔍 First few features: {feature_names[:5]}")

# Define input model (simplified - using your actual feature names)
class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract_encoded: int
    payment_risk: int
    total_services: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 75.5,
                "contract_encoded": 2,
                "payment_risk": 3,
                "total_services": 3
            }
        }

@app.get("/")
def root():
    return {
        "message": "🔮 Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "features_count": len(feature_names)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict")
def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer
    """
    try:
        # Create dataframe with ALL features (fill missing with 0)
        data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the values we have (using your actual feature names)
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
            color = "red"
        elif probability > 0.3:
            risk = "MEDIUM"
            action = "📧 Send engagement email"
            color = "orange"
        else:
            risk = "LOW"
            action = "✅ Regular maintenance"
            color = "green"
        
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

@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    """
    Predict churn for multiple customers
    """
    try:
        results = []
        for i, customer in enumerate(customers):
            # Create dataframe for this customer
            data = pd.DataFrame(0, index=[0], columns=feature_names)
            
            # Fill values
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
            
            # Predict
            data_scaled = scaler.transform(data)
            prob = model.predict_proba(data_scaled)[0][1]
            
            results.append({
                "customer_id": i,
                "churn_probability": round(float(prob), 4),
                "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
            })
        
        # Calculate summary
        probabilities = [r["churn_probability"] for r in results]
        
        return {
            "success": True,
            "total_customers": len(results),
            "average_risk": round(float(np.mean(probabilities)), 4),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH"),
            "medium_risk_count": sum(1 for r in results if r["risk_level"] == "MEDIUM"),
            "low_risk_count": sum(1 for r in results if r["risk_level"] == "LOW"),
            "predictions": results
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)