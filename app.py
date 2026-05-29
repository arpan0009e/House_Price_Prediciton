import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load your trained model
@st.cache_resource
def load_model():
    # Make sure 'model.pkl' is in the same directory as this script
    return joblib.load("model.pkl")

model = load_model()

# 2. App Header Layout
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 House Price Prediction App")
st.write("Input the house specifications below to estimate its market price.")

st.divider()

# 3. Form Layout with Your 5 Exact Variables
col1, col2 = st.columns(2)

with col1:
    sqft_living = st.number_input("Living Area (sqft)", min_value=100, max_value=20000, value=1800, step=50)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
    floors = st.number_input("Number of Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

with col2:
    bathrooms = st.number_input("Number of Bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.25)
    view = st.slider("View Rating (0 to 4)", min_value=0, max_value=4, value=0, step=1)

st.divider()

# 4. Predict Button and Model Logic
if st.button("Calculate Predicted Price", type="primary"):
    # Exact DataFrame mapping to match your model's X columns
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'sqft_living': sqft_living,
        'view': view
    }])
    
    # Generate prediction from model.pkl
    prediction = model.predict(input_data)
    predicted_price = prediction[0]
    
    # Handle any potential negative predictions gracefully, otherwise format nicely
    if predicted_price < 0:
        st.warning("⚠️ The combination of features selected is outside normal boundaries for a realistic price estimate.")
    else:
        st.success(f"### Estimated Value: ${predicted_price:,.2f}")