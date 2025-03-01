import streamlit as st
import numpy as np
import pickle
import os


# ✅ Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Sepsis Detection AI", page_icon="🩺", layout="centered")

# 🔍 Define Model Path and Google Drive Link
MODEL_PATH = "model/sepsis_model.pkl"
GDRIVE_LINK = "https://drive.google.com/uc?id=10kNDhNeTNPucAlpccctvo4A5d8vbIf4g"

# 🔽 Install gdown if not available
try:
    import gdown
except ModuleNotFoundError:
    st.warning("📦 Installing missing package: gdown...")
    os.system("pip install gdown")
    import gdown

# 🔽 Download the model if not present
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    os.makedirs("model", exist_ok=True)
    gdown.download(GDRIVE_LINK, MODEL_PATH, quiet=False)

# 🚨 Check if model exists after download
if not os.path.exists(MODEL_PATH):
    st.error("🚨 Model file not found! Please manually download it from [Google Drive](https://drive.google.com/file/d/10kNDhNeTNPucAlpccctvo4A5d8vbIf4g/view?usp=drive_link) and place it in the `model/` directory.")
    st.stop()

# 📂 Load the trained model
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"⚠ Error loading the model: {e}")
    st.stop()

# 🎯 Streamlit App Layout
st.title("🩺 Sepsis Detection AI")
st.write("This app predicts the *risk of Sepsis* based on patient health metrics.")

# 🏥 **User Inputs**
st.subheader("Enter Patient Data")

HR = st.number_input("Heart Rate (HR)", min_value=30, max_value=200, value=75)
O2Sat = st.number_input("Oxygen Saturation (O2Sat)", min_value=0, max_value=100, value=98)
Temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
SBP = st.number_input("Systolic Blood Pressure (SBP)", min_value=50, max_value=250, value=120)
DBP = st.number_input("Diastolic Blood Pressure (DBP)", min_value=30, max_value=150, value=80)
Resp = st.number_input("Respiratory Rate", min_value=5, max_value=40, value=18)

# Convert Inputs to NumPy Array (Ensure Correct Shape)
features = np.array([HR, O2Sat, Temp, SBP, DBP, Resp]).reshape(1, -1)

# 🏥 **Prediction Button**
if st.button("🔍 Predict Sepsis Risk"):
    try:
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("⚠ *High Sepsis Risk Detected!* Immediate medical attention is required.")
        else:
            st.success("✅ *Low Sepsis Risk.* No immediate danger detected.")
    except Exception as e:
        st.error(f"⚠ Prediction error: {e}")

# 🔗 Footer
st.write("---")
st.write("📌 *Developed by Satya Prakash Shandilya*")
