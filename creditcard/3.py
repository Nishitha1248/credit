import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL & ENCODER
# -----------------------------
try:
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("💳 Fraud Detection System")
st.write("Enter the Transaction Details Below")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

gender_display = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Check for Fraud"):

    if merchant and category and cc_num:

        try:
            merchant = merchant.strip()
            category = category.strip()

            # Convert Gender
            gender = "M" if gender_display == "Male" else "F"

            # Encode last 4 digits of card
            cc_encoded = int(cc_num[-4:]) if cc_num.isdigit() else 0

            # Create DataFrame
            input_data = pd.DataFrame([{
                "merchant": merchant,
                "category": category,
                "amt": amt,
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

            # Final column order (IMPORTANT: must match training)
            input_data = input_data[[
                "merchant",
                "category",
                "amt",
                "gender",
                "cc_num"
            ]]

            # Prediction
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Please Fill All Required Fields")