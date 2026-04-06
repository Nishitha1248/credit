import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
try:
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -------------------------
# TITLE
# -------------------------
st.title("💳 AI-Powered Fraud Detection System")
st.markdown("### Smart Transaction Risk Analyzer")

st.markdown("---")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("📝 Enter Transaction Details")

merchant = st.sidebar.text_input("Merchant Name")
category = st.sidebar.text_input("Category")
amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%.2f")

lat = st.sidebar.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
lon = st.sidebar.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, format="%.6f")

merch_lat = st.sidebar.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
merch_lon = st.sidebar.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, format="%.6f")

hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
day = st.sidebar.slider("Transaction Day", 1, 31, 15)
month = st.sidebar.slider("Transaction Month", 1, 12, 6)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
cc_num = st.sidebar.text_input("Credit Card Number")

st.markdown("---")

# -------------------------
# BUTTON
# -------------------------
if st.sidebar.button("🔍 Analyze Transaction"):

    if merchant and category and cc_num:

        try:
            merchant = merchant.strip()
            category = category.strip()

            # Distance calculation
            distance = geodesic((lat, lon) (merch_lat, merch_lon)).km

            # Encode credit card safely
            cc_encoded = int(cc_num[-4:]) if cc_num.isdigit() else 0

            input_data = pd.DataFrame([{
                "merchant": merchant,
                "category": category,
                "amt": amt,
                "distance": distance,
                "hour": hour,
                "day": day,
                "month": month,
                "gender": gender,
                "cc_num": cc_encoded
            }])

            # Encode categorical values
            for col in ["merchant", "category", "gender"]:
                value = input_data[col].iloc[0]
                if value in encoder[col].classes_:
                    input_data[col] = encoder[col].transform(input_data[col])
                else:
                    input_data[col] = 0

            # Correct column order
            input_data = input_data[[
                "merchant", "category", "amt", "distance",
                "hour", "day", "month", "gender", "cc_num"
            ]]

            # Prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            # -------------------------
            # DISPLAY RESULTS
            # -------------------------

            st.subheader("📊 Transaction Analysis Result")

            col1, col2, col3 = st.columns(3)

            col1.metric("💰 Amount", f"${amt}")
            col2.metric("📍 Distance (km)", f"{round(distance,2)} km")
            col3.metric("⚠ Fraud Probability", f"{round(probability,2)} %")

            st.markdown("---")

            if probability < 30:
                st.success("🟢 LOW RISK - Legitimate Transaction")
            elif probability < 70:
                st.warning("🟠 MEDIUM RISK - Needs Review")
            else:
                st.error("🔴 HIGH RISK - Fraudulent Transaction")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.warning("⚠ Please Fill All Required Fields")