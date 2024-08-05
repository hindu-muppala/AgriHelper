import streamlit as st
import joblib
import numpy as np

# Load the saved model, scaler, and label encoders
model = joblib.load('crop_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_state = joblib.load('label_encoder_state.pkl')
label_encoder_district = joblib.load('label_encoder_district.pkl')

# Function to make predictions
def predict_crop(N, P, K, temp, humid, ph, rainfal, investment, state_name, district_name):
    # Encode categorical inputs
    state_name_encoded = label_encoder_state.transform([state_name])[0]
    district_name_encoded = label_encoder_district.transform([district_name])[0]
    
    # Prepare input features
    features = np.array([[N, P, K, temp, humid, ph, rainfal, investment, state_name_encoded, district_name_encoded]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit UI
st.title("Crop Prediction Model")

N = st.number_input("Enter Nitrogen (N)", min_value=0.0)
P = st.number_input("Enter Phosphorus (P)", min_value=0.0)
K = st.number_input("Enter Potassium (K)", min_value=0.0)
temp = st.number_input("Enter Temperature (°C)", min_value=-50.0, max_value=50.0)
humid = st.number_input("Enter Humidity (%)", min_value=0, max_value=100)
ph = st.number_input("Enter pH level", min_value=0.0, max_value=14.0)
rainfal = st.number_input("Enter Rainfall (mm)", min_value=0.0)
investment = st.number_input("Enter Investment (₹)", min_value=0.0)
state_name = st.selectbox("Select State", options=label_encoder_state.classes_)  # Assuming state is a string
district_name = st.selectbox("Select District", options=label_encoder_district.classes_)  # Assuming district is a string

if st.button("Predict Crop"):
    try:
        crop = predict_crop(N, P, K, temp, humid, ph, rainfal, investment, state_name, district_name)
        st.write(f"The predicted crop is: {crop}")
    except Exception as e:
        st.error(f"Error: {e}")
