
import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

try:
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoder.jb")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


st.title("💳 Fraud Detection System")
st.write("Enter the Transaction Details Below")



merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")


lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.6f")


merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
merch_lon = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, format="%.6f")



hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)



gender_display = st.selectbox("Gender", ["Male", "Female"])


cc_num = st.text_input("Credit Card Number")

if st.button("Check for Fraud"):

    if merchant and category and cc_num:

        try:
            
            merchant = merchant.strip()
            category = category.strip()

            
            gender = "M" if gender_display == "Male" else "F"

            
            distance = haversine(lat, lon, merch_lat, merch_lon)

           
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


            
            for col in ["merchant", "category", "gender"]:
                value = input_data[col].iloc[0]

                if value in encoder[col].classes_:
                    input_data[col] = encoder[col].transform(input_data[col])
                else:
                    input_data[col] = 0  

            
            input_data = input_data[[
                "merchant",
                "category",
                "amt",
                "distance",
                "hour",
                "day",
                "month",
                "gender",
                "cc_num"
            ]]

           
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.error("Please Fill All Required Fields")