#CHURN_NEW/deployment/simple_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import uvicorn
from pathlib import Path

app = FastAPI(title="Churn Prediction API")

# Try all possible locations
possible_locations = [
    Path("../logistic_regression_model.pkl"),           # Parent folder
    Path("../notebooks/logistic_regression_model.pkl"), # Notebooks folder
    Path("./logistic_regression_model.pkl"),            # Current folder
    Path("C:/Users/HEMANATH/Desktop/churn_new/logistic_regression_model.pkl"),  # Absolute path
    Path("C:/Users/HEMANATH/Desktop/churn_new/notebooks/logistic_regression_model.pkl"),
]

# Load models
model = None
scaler = None
feature_names = None

print("Searching for model files...")

for loc in possible_locations:
    if loc.exists():
        print(f"Found model at: {loc}")
        model = joblib.load(loc)
        
        # Try to find scaler in same directory
        scaler_loc = loc.parent / "scaler.pkl"
        if scaler_loc.exists():
            scaler = joblib.load(scaler_loc)
            print(f"Found scaler at: {scaler_loc}")
        
        # Try to find features
        features_loc = loc.parent / "feature_names.pkl"
        if features_loc.exists():
            feature_names = joblib.load(features_loc)
            print(f"Found features at: {features_loc}")
        break

if model is None:
    print("\nERROR: Could not find model files!")
    print("Please run the find_files.py script first to locate your files.")
    print("Or manually specify the correct path.")
    exit(1)

print(f"\nAll models loaded! Using {len(feature_names)} features.")

# Define input
class CustomerInput(BaseModel):
    tenure: int
    monthly_charges: float
    contract_encoded: int
    payment_risk: int
    total_services: int

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "features_loaded": len(feature_names) if feature_names else 0
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    # Create dataframe with ALL features
    data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the values we have
    data['tenure'] = customer.tenure
    data['MonthlyCharges'] = customer.monthly_charges
    data['Contract_encoded'] = customer.contract_encoded
    data['PaymentRisk'] = customer.payment_risk
    data['TotalServices'] = customer.total_services
    
    # Scale and predict
    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]
    
    # Determine risk
    if prob > 0.7:
        risk = "HIGH"
        action = "Call customer with offer"
    elif prob > 0.3:
        risk = "MEDIUM"
        action = "Send discount email"
    else:
        risk = "LOW"
        action = "Regular check-in"
    
    return {
        "churn_probability": round(float(prob), 3),
        "risk_level": risk,
        "action": action,
        "features_used": len(feature_names)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)