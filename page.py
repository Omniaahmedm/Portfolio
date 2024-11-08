import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# Streamlit interface
st.title("Suspicious Activity Detection")

st.write(
    "This application detects suspicious activity based on network traffic data. Enter the relevant information to predict whether the activity is suspicious or not."
)

# Collect user input for prediction
total_bytes_in = st.number_input('Total Bytes In', min_value=0, step=1)
total_bytes_out = st.number_input('Total Bytes Out', min_value=0, step=1)
failed_requests = st.number_input('Failed Requests (404, 403)', min_value=0, step=1)
request_count = st.number_input('Request Count', min_value=0, step=1)
suspicious_country = st.selectbox('Suspicious Country?', ['No', 'Yes'])
exfiltration_suspicious = st.selectbox('Exfiltration Suspicious?', ['No', 'Yes'])
requests_per_hour = st.number_input('Requests per Hour', min_value=0, step=1)
brute_force_or_dos = st.selectbox('Brute Force or DoS?', ['No', 'Yes'])

# Convert user input to numerical values (0 or 1 for categorical inputs)
suspicious_country = 1 if suspicious_country == 'Yes' else 0
exfiltration_suspicious = 1 if exfiltration_suspicious == 'Yes' else 0
brute_force_or_dos = 1 if brute_force_or_dos == 'Yes' else 0

# Create a DataFrame with user input
input_data = pd.DataFrame({
    'total_bytes_in': [total_bytes_in],
    'total_bytes_out': [total_bytes_out],
    'failed_requests': [failed_requests],
    'request_count': [request_count],
    'suspicious_country': [suspicious_country],
    'exfiltration_suspicious': [exfiltration_suspicious],
    'requests_per_hour': [requests_per_hour],
    'brute_force_or_dos': [brute_force_or_dos]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("⚠️ **Suspicious Activity Detected!** ⚠️")
    else:
        st.write("✅ **No Suspicious Activity.** ✅")


