import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("fraud_small_model.joblib")
    encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.title("💳 Fraud Detection System")
st.write("Simple Fraud Checker")


merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0)
gender_display = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

if st.button("Check for Fraud"):

    if merchant and category and cc_num:

        try:
            gender = "M" if gender_display == "Male" else "F"
            cc_encoded = int(cc_num[-4:]) if cc_num.isdigit() else 0

            input_data = pd.DataFrame([{
                "merchant": merchant.strip(),
                "category": category.strip(),
                "amt": amt,
                "gender": gender,
                "cc_num": cc_encoded
            }])

            # Encode
            for col in ["merchant", "category", "gender"]:
                if input_data[col][0] in encoder[col].classes_:
                    input_data[col] = encoder[col].transform(input_data[col])
                else:
                    input_data[col] = 0

            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 Fraud Detected!")
            else:
                st.success("✅ Legit Transaction")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please fill all fields")