# 📁 CHURN_NEW_2/deployment/simple_dashboard.py
# Streamlit dashboard that works with your files

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    """Load models from notebooks folder"""
    try:
        notebooks_path = Path("../notebooks")
        
        model = joblib.load(notebooks_path / "logistic_regression_model.pkl")
        scaler = joblib.load(notebooks_path / "scaler.pkl")
        feature_names = joblib.load(notebooks_path / "feature_names.pkl")
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, feature_names = load_models()

if model is None:
    st.error("❌ Could not load models. Check if files exist in ../notebooks/")
    st.stop()

st.success(f"✅ Model loaded successfully! Using {len(feature_names)} features")

# Sidebar
with st.sidebar:
    st.header("🎯 Customer Details")
    
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
    
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    contract_encoded = contract_map[contract]
    
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    payment_map = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer": 1,
        "Credit card": 1
    }
    payment_risk = payment_map[payment]
    
    total_services = st.slider("Total Services", 1, 8, 3)
    
    predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Information")
    
    info_df = pd.DataFrame({
        "Feature": ["Tenure", "Monthly Charges", "Contract", "Payment Method", "Total Services"],
        "Value": [f"{tenure} months", f"${monthly_charges}", contract, payment, total_services]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🔮 Prediction Result")
    
    if predict_button:
        # Create feature vector
        data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill values
        if 'tenure' in feature_names:
            data['tenure'] = tenure
        if 'MonthlyCharges' in feature_names:
            data['MonthlyCharges'] = monthly_charges
        if 'Contract_encoded' in feature_names:
            data['Contract_encoded'] = contract_encoded
        if 'PaymentRisk' in feature_names:
            data['PaymentRisk'] = payment_risk
        if 'TotalServices' in feature_names:
            data['TotalServices'] = total_services
        
        # Predict
        data_scaled = scaler.transform(data)
        probability = model.predict_proba(data_scaled)[0][1]
        
        # Display result
        col_prob, col_gauge = st.columns(2)
        
        with col_prob:
            if probability > 0.7:
                st.error(f"### 🚨 HIGH RISK")
                st.markdown(f"## {probability:.1%}")
                st.info("**Action:** Immediate retention offer needed")
            elif probability > 0.3:
                st.warning(f"### ⚠️ MEDIUM RISK")
                st.markdown(f"## {probability:.1%}")
                st.info("**Action:** Send engagement email")
            else:
                st.success(f"### ✅ LOW RISK")
                st.markdown(f"## {probability:.1%}")
                st.info("**Action:** Regular maintenance")
        
        with col_gauge:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.markdown("---")
st.subheader("📊 Feature Importance")

# Get feature importance (for linear models)
if hasattr(model, 'coef_'):
    importances = np.abs(model.coef_[0])
    importances = importances / importances.sum()
    
    # Get top 10 features
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    fig = px.bar(
        feat_imp,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Feature Importances',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Sample predictions
st.markdown("---")
st.subheader("📈 Sample Predictions by Tenure")

# Generate predictions for different tenures
tenures = range(0, 73, 6)
risks = []

for t in tenures:
    data = pd.DataFrame(0, index=[0], columns=feature_names)
    if 'tenure' in feature_names:
        data['tenure'] = t
    if 'MonthlyCharges' in feature_names:
        data['MonthlyCharges'] = 70
    if 'Contract_encoded' in feature_names:
        data['Contract_encoded'] = 2
    if 'PaymentRisk' in feature_names:
        data['PaymentRisk'] = 3
    if 'TotalServices' in feature_names:
        data['TotalServices'] = 3
    
    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]
    risks.append(prob)

# Plot
fig = px.line(
    x=tenures,
    y=risks,
    markers=True,
    title='Churn Risk by Tenure',
    labels={'x': 'Tenure (months)', 'y': 'Churn Probability'}
)
fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🔮 **Churn Prediction Dashboard** | Model: Logistic Regression")