# app.py
import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load your trained model
model = joblib.load('JaizBank_ProvantageNGX-AI_model.pkl')

# App title and logo
st.set_page_config(page_title="ProvantageNGX AI", layout="centered")

# Load logo
logo = Image.open("static/logo.png")
st.image(logo, width=200, caption='PV')

st.title("📈Jaiz Bank Stock Price Prediction")
st.markdown("Enter the values below to predict the **next day's stock price**.")

# Input form
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        price = st.text_input("Closing Price", value="2.00")
        ma_5 = st.text_input("5-Day Moving Average", value="2.01")
        lag_1 = st.text_input("Previous Day Price (Lag_1)", value="2.02")

    with col2:
        ma_10 = st.text_input("10-Day Moving Average", value="2.00")
        lag_2 = st.text_input("Price from 2 Days Ago (Lag_2)", value="2.03")
        daily_return = st.text_input("Daily Return (%)", value="0.5")

    submit_button = st.form_submit_button(label='🔮 Predict')

# Prediction
if submit_button:
    try:
        features = np.array([
            float(price),
            float(ma_5),
            float(ma_10),
            float(lag_1),
            float(lag_2),
            float(daily_return)
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        st.success(f"📊 **Predicted Stock Price for the Next Day: ₦{prediction:.4f}**")
    except Exception as e:
        st.error(f"Error: {e}")
