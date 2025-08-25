import streamlit as st
import requests
import os  # <-- import os to access environment variables

st.title("🔮 Feedforward Neural Network Demo")
st.write("Enter 4 features and get a prediction from the trained model.")

# Get backend URL from environment variable, fallback to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

features = []
for i in range(4):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

if st.button("Predict"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",  # <-- use the environment variable here
            json={"features": features}
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted class: {result['predicted_class']}")
            st.write(f"Probabilities: {result['probabilities']}")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
