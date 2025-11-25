import streamlit as st
import requests

st.set_page_config(page_title="NSL-KDD Intrusion Detection", layout="centered")
st.title("NSL-KDD Intrusion Detection")
st.markdown("Provide a comma-separated feature vector in the same order the model was trained on.")

API_URL = st.text_input("API URL", "http://localhost:8000/predict")
features_input = st.text_area("Feature vector (comma separated)", height=120)

if st.button("Predict"):
    if not features_input.strip():
        st.error("Please enter a feature vector")
    else:
        try:
            features = [float(x.strip()) for x in features_input.split(",") if x.strip() != ""]
            r = requests.post(API_URL, json={"features": features})
            if r.status_code == 200:
                st.success("Prediction returned")
                st.json(r.json())
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(str(e))
