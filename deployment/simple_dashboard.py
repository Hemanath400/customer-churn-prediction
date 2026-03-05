# 📁 CHURN_NEW/deployment/simple_dashboard.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff4d4d;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffa64d;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #4CAF50;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Customer Churn Predictor</h1>', unsafe_allow_html=True)

# ============================================
# Load Models
# ============================================

@st.cache_resource
def load_models():
    """Load trained models from notebooks folder"""
    try:
        model_path = Path("../notebooks/logistic_regression_model.pkl")
        scaler_path = Path("../notebooks/scaler.pkl")
        features_path = Path("../notebooks/feature_names.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        
        st.sidebar.success(f"✅ Model loaded with {len(feature_names)} features")
        return model, scaler, feature_names
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None, None, None

model, scaler, feature_names = load_models()

# ============================================
# Sidebar - Customer Inputs
# ============================================

with st.sidebar:
    st.header("📝 Customer Details")
    st.markdown("---")
    
    # Basic Info
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1, 
                                 help="How long the customer has been with the company")
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=70.0, step=0.01)
    
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    st.markdown("---")
    
    # Services
    st.subheader("Services")
    col1, col2 = st.columns(2)
    with col1:
        total_services = st.slider("Total Services", 1, 8, 3, 
                                   help="Number of services subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with col2:
        premium_services = st.slider("Premium Services", 0, 6, 1,
                                     help="Number of premium services (security, streaming)")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    st.markdown("---")
    
    # Contract & Payment
    st.subheader("Contract & Payment")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    # Family
    st.subheader("Family")
    col1, col2 = st.columns(2)
    with col1:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with col2:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

# ============================================
# Main Area - Predictions
# ============================================

if predict_button and model is not None:
    
    # ========================================
    # STEP 1: Calculate ALL 16 features correctly
    # ========================================
    
    # 1. Basic features
    total_charges = monthly_charges * tenure
    
    # 2. Encoding maps
    binary_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}
    
    # 3. Contract encoding (0-2 scale)
    if contract == "Month-to-month":
        contract_encoded = 2
        contract_risk = 2  # Highest risk
    elif contract == "One year":
        contract_encoded = 1
        contract_risk = 1  # Medium risk
    else:  # Two year
        contract_encoded = 0
        contract_risk = 0  # Lowest risk
    
    # 4. Payment risk (1-3 scale, higher = riskier)
    payment_risk_map = {
        "Electronic check": 3,  # Riskiest
        "Mailed check": 2,
        "Bank transfer": 1,
        "Credit card": 1        # Safest
    }
    payment_risk = payment_risk_map[payment]
    
    # 5. Internet service encoding
    internet_map = {"Fiber optic": 2, "DSL": 1, "No": 0}
    internet_encoded = internet_map[internet]
    
    # 6. CRITICAL: RiskScore - This MUST be high for new customers!
    # New customers (low tenure) should have HIGH risk score
    risk_score = 0.0
    
    # Tenure risk (NEW CUSTOMERS = HIGH RISK)
    if tenure < 6:
        risk_score += 0.5  # Very new = very risky
    elif tenure < 12:
        risk_score += 0.3  # New = risky
    elif tenure < 24:
        risk_score += 0.2  # Established = moderate
    else:
        risk_score += 0.1  # Loyal = low risk
    
    # Contract risk
    if contract == "Month-to-month":
        risk_score += 0.3
    elif contract == "One year":
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Payment risk
    if payment == "Electronic check":
        risk_score += 0.2
    elif payment == "Mailed check":
        risk_score += 0.1
    
    # Normalize to 0-1 scale
    risk_score = min(risk_score, 1.0)
    
    # ========================================
    # STEP 2: Create DataFrame with ALL features
    # ========================================
    
    # Create empty dataframe with correct columns
    data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill ALL 16 features
    feature_values = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': binary_map[senior_citizen],
        'TotalServices': total_services,
        'PremiumServices': premium_services,
        'ContractRisk': contract_risk,
        'PaymentRisk': payment_risk,
        'RiskScore': risk_score,  # This is CRITICAL - must be high for new customers!
        'gender_encoded': gender_map[gender],
        'Partner_encoded': binary_map[partner],
        'Dependents_encoded': binary_map[dependents],
        'PhoneService_encoded': binary_map[phone_service],
        'PaperlessBilling_encoded': binary_map[paperless],
        'InternetService_encoded': internet_encoded,
        'Contract_encoded': contract_encoded
    }
    
    for col, value in feature_values.items():
        if col in data.columns:
            data[col] = value
    
    # ========================================
    # STEP 3: Debug - Show what's being sent
    # ========================================
    
    with st.expander("🔍 Debug: Feature Values Sent to Model"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Risk-Related Features:**")
            st.write(f"- Tenure: {tenure} months")
            st.write(f"- ContractRisk: {contract_risk} (0-2 scale)")
            st.write(f"- PaymentRisk: {payment_risk} (1-3 scale)")
            st.write(f"- RiskScore: {risk_score:.2f} (0-1 scale)")
        
        with col2:
            st.write("**Encoded Features:**")
            st.write(f"- Contract_encoded: {contract_encoded}")
            st.write(f"- InternetService_encoded: {internet_encoded}")
            st.write(f"- Payment Method: {payment}")
    
    # ========================================
    # STEP 4: Make Prediction
    # ========================================
    
    # Scale features
    data_scaled = scaler.transform(data)
    probability = model.predict_proba(data_scaled)[0][1]
    
    # ========================================
    # STEP 5: Display Results
    # ========================================
    
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Big probability display
        if probability > 0.7:
            st.markdown(f"""
            <div class="risk-high">
                <h1>🚨 HIGH RISK</h1>
                <h2>{probability:.1%}</h2>
                <p>Immediate retention offer needed!</p>
            </div>
            """, unsafe_allow_html=True)
            action = "📞 Call customer with special offer"
            color = "red"
        elif probability > 0.3:
            st.markdown(f"""
            <div class="risk-medium">
                <h1>⚠️ MEDIUM RISK</h1>
                <h2>{probability:.1%}</h2>
                <p>Monitor and send engagement email</p>
            </div>
            """, unsafe_allow_html=True)
            action = "📧 Send discount offer via email"
            color = "orange"
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h1>✅ LOW RISK</h1>
                <h2>{probability:.1%}</h2>
                <p>Regular maintenance</p>
            </div>
            """, unsafe_allow_html=True)
            action = "✨ Regular check-in"
            color = "green"
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Recommended Action:** {action}")
    
    # ========================================
    # STEP 6: Risk Factor Breakdown
    # ========================================
    
    st.markdown("---")
    st.subheader("📈 Risk Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk factors table
        risk_factors = pd.DataFrame({
            'Factor': ['Tenure', 'Contract', 'Payment Method', 'Internet Service'],
            'Risk Level': [
                '🔴 HIGH' if tenure < 12 else '🟢 LOW',
                '🔴 HIGH' if contract == 'Month-to-month' else '🟢 LOW',
                '🔴 HIGH' if payment == 'Electronic check' else '🟡 MEDIUM' if payment == 'Mailed check' else '🟢 LOW',
                '🟡 MEDIUM' if internet == 'Fiber optic' else '🟢 LOW'
            ],
            'Impact': [0.4, 0.3, 0.2, 0.1]
        })
        
        fig = px.bar(risk_factors, x='Impact', y='Factor', orientation='h',
                    color='Risk Level', color_discrete_map={
                        '🔴 HIGH': 'red', '🟡 MEDIUM': 'orange', '🟢 LOW': 'green'
                    })
        fig.update_layout(title="Risk Factor Contribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        if hasattr(model, 'coef_'):
            # For logistic regression, get top features
            feature_importance = pd.DataFrame({
                'Feature': feature_names[:10],
                'Importance': np.abs(model.coef_[0][:10])
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature',
                        orientation='h', color='Importance',
                        color_continuous_scale='reds')
            fig.update_layout(title="Top 10 Feature Importance", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ========================================
# Show sample analysis even without prediction
# ========================================

else:
    st.info("👈 Enter customer details and click 'Predict Churn' to see results")
    
    # Show sample visualization
    st.subheader("📊 Sample: Churn Probability by Tenure")
    
    # Generate sample data
    tenures = list(range(0, 73, 6))
    risks = []
    
    for t in tenures:
        # Simulate realistic curve - HIGH for new, LOW for loyal
        if t < 6:
            risk = 0.85
        elif t < 12:
            risk = 0.70
        elif t < 24:
            risk = 0.45
        elif t < 48:
            risk = 0.25
        else:
            risk = 0.12
        risks.append(risk)
    
    fig = px.line(x=tenures, y=risks, markers=True,
                  labels={'x': 'Tenure (months)', 'y': 'Churn Probability'})
    fig.update_traces(line=dict(color='red', width=3))
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Model trained with {len(feature_names) if feature_names is not None else 16} features | Dashboard v3.0")