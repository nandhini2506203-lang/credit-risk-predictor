import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# FILE PATH SETUP
# -------------------------------
import joblib

model = joblib.load("logistic_model.pkl")
# -------------------------------
# LOAD MODEL SAFELY
# -------------------------------
try:
    model = joblib.load("logistic_model.pkl")
except Exception as e:
    st.error(f"❌ Model not found or error loading model: {e}")
    st.stop()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Loan Risk Prediction System")
st.markdown("Enter customer details to predict loan risk")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📊 Customer Financial Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    per_capita_income = st.number_input("Per Capita Income", value=50000)
    yearly_income = st.number_input("Yearly Income", value=500000)
    total_debt = st.number_input("Total Debt", value=100000)

with col2:
    credit_score = st.number_input("Credit Score", value=700)
    num_credit_cards = st.number_input("Number of Credit Cards", value=2)
    amount = st.number_input("Transaction Amount", value=20000)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Loan Risk"):

    try:
        input_data = pd.DataFrame([{
            'current_age': age,
            'per_capita_income': per_capita_income,
            'yearly_income': yearly_income,
            'total_debt': total_debt,
            'credit_score': credit_score,
            'num_credit_cards': num_credit_cards,
            'amount': amount
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📢 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk ({probability:.2%} probability)")

            st.markdown("### 📌 Recommendations:")
            st.write("✔ Reduce debt-to-income ratio")
            st.write("✔ Improve credit score (pay EMIs on time)")
            st.write("✔ Avoid multiple credit cards")
            st.write("✔ Increase stable income")

        else:
            st.success(f"✅ Low Risk ({probability:.2%} probability)")

            st.markdown("### 📌 Recommendations:")
            st.write("✔ Maintain good repayment history")
            st.write("✔ Keep credit utilization low")
            st.write("✔ Continue stable income flow")
            st.write("✔ Use credit responsibly")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
