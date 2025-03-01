import streamlit as st
import requests

st.title("Sepsis Detection AI")
features = st.text_input("Enter patient vitals (comma-separated)")

if st.button("Predict Sepsis Risk"):
    features_list = [float(i) for i in features.split(",")]
    response = requests.post("http://127.0.0.1:5000/predict", json={"features": features_list})
    result = response.json()
    st.write(f"Sepsis Risk: {'High' if result['sepsis_risk'] == 1 else 'Low'}")

