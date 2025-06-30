import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ----------------------------
# Cargar modelo entrenado
# ----------------------------
model_pipeline = joblib.load("modelo_completo.joblib")

st.title("🚨 Predicción de Churn de Clientes")

st.markdown("""
Este modelo predice si un cliente está en riesgo de abandonar (churn).
Completa la información del cliente para obtener una predicción.
""")

# ----------------------------
# Crear formulario de entrada
# ----------------------------
with st.form("prediction_form"):
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    under_30 = st.selectbox("Under 30", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    number_of_dependents = st.number_input("Número de dependientes", min_value=0, max_value=10, value=0)
    referred_a_friend = st.selectbox("Referred a friend", ["Yes", "No"])
    number_of_referrals = st.number_input("Número de referidos", min_value=0, max_value=10, value=0)
    tenure_in_months = st.number_input("Tenure (meses)", min_value=0, max_value=100, value=12)
    offer = st.selectbox("Oferta", ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E", "no_offer"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    avg_monthly_long_distance_charges = st.number_input("Avg monthly long distance charges", min_value=0.0, value=10.0)
    multiple_lines = st.selectbox("Multiple lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["Yes", "No"])
    internet_type = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "Cable", "No"])
    avg_monthly_gb_download = st.number_input("Avg monthly GB download", min_value=0.0, value=20.0)
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection_plan = st.selectbox("Device Protection Plan", ["Yes", "No"])
    premium_tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
    unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-Month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
    total_refunds = st.number_input("Total Refunds", min_value=0.0, value=0.0)
    total_extra_data_charges = st.number_input("Total Extra Data Charges", min_value=0.0, value=0.0)
    total_long_distance_charges = st.number_input("Total Long Distance Charges", min_value=0.0, value=0.0)
    total_revenue = st.number_input("Total Revenue", min_value=0.0, value=1100.0)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    cltv = st.number_input("CLTV", min_value=0, value=5000)

    submitted = st.form_submit_button("Predecir Churn")

# ----------------------------
# Predicción
# ----------------------------
if submitted:
    # Crear dataframe con una sola fila
    input_dict = {
        "gender": [gender],
        "age": [age],
        "under_30": [under_30],
        "senior_citizen": [senior_citizen],
        "married": [married],
        "dependents": [dependents],
        "number_of_dependents": [number_of_dependents],
        "referred_a_friend": [referred_a_friend],
        "number_of_referrals": [number_of_referrals],
        "tenure_in_months": [tenure_in_months],
        "offer": [offer],
        "phone_service": [phone_service],
        "avg_monthly_long_distance_charges": [avg_monthly_long_distance_charges],
        "multiple_lines": [multiple_lines],
        "internet_service": [internet_service],
        "internet_type": [internet_type],
        "avg_monthly_gb_download": [avg_monthly_gb_download],
        "online_security": [online_security],
        "online_backup": [online_backup],
        "device_protection_plan": [device_protection_plan],
        "premium_tech_support": [premium_tech_support],
        "streaming_tv": [streaming_tv],
        "streaming_movies": [streaming_movies],
        "streaming_music": [streaming_music],
        "unlimited_data": [unlimited_data],
        "contract": [contract],
        "paperless_billing": [paperless_billing],
        "payment_method": [payment_method],
        "monthly_charge": [monthly_charge],
        "total_charges": [total_charges],
        "total_refunds": [total_refunds],
        "total_extra_data_charges": [total_extra_data_charges],
        "total_long_distance_charges": [total_long_distance_charges],
        "total_revenue": [total_revenue],
        "satisfaction_score": [satisfaction_score],
        "cltv": [cltv],
    }

    input_df = pd.DataFrame(input_dict)

    # Predecir
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

    # Mostrar resultado
    if prediction == 1:
        st.error(f"⚠️ Este cliente probablemente hará churn. Probabilidad: {prediction_proba:.2f}")
    else:
        st.success(f"✅ Este cliente probablemente NO hará churn. Probabilidad: {prediction_proba:.2f}")
