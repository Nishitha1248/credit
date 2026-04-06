import streamlit as st
import pandas as pd
import joblib

# Load model
try:
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.title("💳 Fraud Detection System")

# Inputs
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0)
gender_display = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

if st.button("Check for Fraud"):

    if merchant and category and cc_num:

        gender = "M" if gender_display == "Male" else "F"
        cc_encoded = int(cc_num[-4:]) if cc_num.isdigit() else 0

        input_data = pd.DataFrame([{
            "merchant": merchant.strip(),
            "category": category.strip(),
            "amt": amt,
            "gender": gender,
            "cc_num": cc_encoded
        }])

        # Encode safely
        for col in ["merchant", "category", "gender"]:
            value = input_data[col].iloc[0]

            if value in encoder[col].classes_:
                input_data[col] = encoder[col].transform(input_data[col])
            else:
                input_data[col] = 0   # unknown value

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraud Detected! Risk: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction (Risk: {prob:.2f})")

    else:
        st.warning("⚠️ Please fill all fields")