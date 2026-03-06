# 📁 mlflow_tracking.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import os
from datetime import datetime

print("🚀 Starting MLflow Experiment Tracking...")

# Load your preprocessed data
try:
    # Try loading from notebooks folder
    df = pd.read_csv('notebooks/telco_churn_cleaned.csv')
    print("✅ Loaded data from notebooks/telco_churn_cleaned.csv")
except:
    # Fallback to original data
    df = pd.read_csv('notebooks/Telco-Customer-Churn.csv')
    print("✅ Loaded data from notebooks/Telco-Customer-Churn.csv")

# Prepare features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Handle categorical variables (simple encoding for demo)
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Set MLflow experiment
mlflow.set_experiment("Customer Churn Prediction")

# ============================================
# Experiment 1: Random Forest
# ============================================
print("\n🌲 Training Random Forest...")
with mlflow.start_run(run_name=f"Random_Forest_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log feature importance (top 10)
    feature_names = X_encoded.columns[:10]
    importances = rf_model.feature_importances_[:10]
    for name, imp in zip(feature_names, importances):
        mlflow.log_metric(f"importance_{name}", imp)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    print(f"✅ Random Forest - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

# ============================================
# Experiment 2: Logistic Regression
# ============================================
print("\n📈 Training Logistic Regression...")
with mlflow.start_run(run_name=f"Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    
    # Log parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("test_size", 0.2)
    
    # Train model
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    
    # Predict
    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log coefficients (top 10)
    coefs = lr_model.coef_[0][:10]
    for name, coef in zip(feature_names, coefs):
        mlflow.log_metric(f"coef_{name}", coef)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    
    print(f"✅ Logistic Regression - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

print("\n" + "="*50)
print("🎉 MLflow Tracking Complete!")
print("="*50)
print("\n📊 To view results, run:")
print("   mlflow ui")
print("   Then open http://localhost:5000 in your browser")