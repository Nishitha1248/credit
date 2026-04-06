import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -----------------------------
# Load Model and Encoder
# -----------------------------
try:
    model = joblib.load("fraud_small_model.joblib")
    encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -----------------------------
# Display Image at Top
# -----------------------------

image = Image.open("C:\\Users\\User\\OneDrive\\Desktop\\creditcard\\credit_card_form.png")# Make sure this file is in the same folder
st.image(image, use_column_width=True)
st.title("💳 Fraud Detection System")
st.write("Enter the Transaction Details Below")

# -----------------------------
# Input Fields
# -----------------------------
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0)
gender_display = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Check for Fraud"):

    if merchant and category and cc_num:

        try:
            # Encode gender
            gender = "M" if gender_display == "Male" else "F"

            # Encode full credit card number safely
            cc_encoded = int(cc_num) % 1000000 if cc_num.isdigit() else 0

            # Prepare input dataframe
            input_data = pd.DataFrame([{
                "merchant": merchant.strip(),
                "category": category.strip(),
                "amt": amt,
                "gender": gender,
                "cc_num": cc_encoded
            }])

            # Encode categorical columns
            for col in ["merchant", "category", "gender"]:
                value = input_data[col][0]
                if value in encoder[col].classes_:
                    input_data[col] = encoder[col].transform(input_data[col])
                else:
                    input_data[col] = 0

            # Make prediction
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.warning("Please fill all fields")