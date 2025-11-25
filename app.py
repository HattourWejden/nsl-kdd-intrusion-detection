import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import time
from lime.lime_tabular import LimeTabularExplainer

# ------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Intrusion Detection System – NSL-KDD",
    page_icon="🛡️",
    layout="wide",
)

# ------------------------------------------------------------------
# CUSTOM CSS FOR CLEAN AND LIVELY UI
# ------------------------------------------------------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background-color:#f5f5f5; }
.metric-card { background-color:#ffffff; padding:20px; border-radius:14px; border:1px solid #ccc; margin-bottom:15px; box-shadow:2px 2px 10px rgba(0,0,0,0.1);}
.stButton>button { border-radius:10px; font-size:16px; padding:0.6rem 1.2rem; font-weight:600;}
.predict-btn>button { background-color:#0078ff !important; color:white !important;}
.reset-btn>button { background-color:#ff5252 !important; color:white !important;}
.sample-btn>button { background-color:#444 !important; color:white !important; color:white !important;}
.pred-normal {background:#0e7b29; padding:18px; border-radius:12px; color:white; font-size:22px; font-weight:bold; text-align:center;}
.pred-attack {background:#b00020; padding:18px; border-radius:12px; color:white; font-size:22px; font-weight:bold; text-align:center;}
.progress-bar-container {background-color:#ddd; border-radius:10px; height:30px; margin-top:10px;}
.progress-bar-fill {height:100%; border-radius:10px; text-align:center; color:white; font-weight:bold; line-height:30px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD MODEL, SCALER AND COLUMNS
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/nslkdd_dnn_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.save")

@st.cache_resource
def load_columns():
    return joblib.load("models/columns.save")

model = load_model()
scaler = load_scaler()
model_columns = load_columns()

# ------------------------------------------------------------------
# SIDEBAR — SAMPLE INPUTS
# ------------------------------------------------------------------
st.sidebar.title("🔍 Test Samples")
st.sidebar.write("Choose a predefined example to auto-fill the form:")

sample_inputs = {
    "Normal": {
        "src_bytes": 215, "dst_bytes": 4500, "logged_in": 1,
        "count": 5, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "same_srv_rate": 0.7, "diff_srv_rate": 0.1,
        "dst_host_count": 30, "dst_host_srv_count": 15,
        "dst_host_same_srv_rate": 0.8, "dst_host_diff_srv_rate": 0.05,
        "dst_host_same_src_port_rate": 0.4, "dst_host_srv_diff_host_rate": 0.02,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "service": "http", "flag": "SF"
    },
    "DoS Attack": {
        "src_bytes": 0, "dst_bytes": 0, "logged_in": 0,
        "count": 200, "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "same_srv_rate": 0.0, "diff_srv_rate": 1.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "service": "http", "flag": "S0"
    },
    "Probe Attack": {
        "src_bytes": 300, "dst_bytes": 50, "logged_in": 0,
        "count": 20, "serror_rate": 0.2, "srv_serror_rate": 0.3,
        "same_srv_rate": 0.3, "diff_srv_rate": 0.5,
        "dst_host_count": 100, "dst_host_srv_count": 30,
        "dst_host_same_srv_rate": 0.4, "dst_host_diff_srv_rate": 0.4,
        "dst_host_same_src_port_rate": 0.3, "dst_host_srv_diff_host_rate": 0.4,
        "dst_host_serror_rate": 0.1, "dst_host_srv_serror_rate": 0.1,
        "service": "smtp", "flag": "REJ"
    }
}

selected_sample = st.sidebar.selectbox("Select example", ["None"] + list(sample_inputs.keys()))
if selected_sample != "None":
    form_data = sample_inputs[selected_sample]
    st.sidebar.info(f"Example loaded: {selected_sample}")
else:
    form_data = sample_inputs["Normal"]

# ------------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------------
st.title("🛡️ Intrusion Detection System – NSL-KDD")
st.markdown("### Detect whether a network connection is **Normal** or an **Attack**")

# ------------------------------------------------------------------
# FEATURE INPUT FORM
# ------------------------------------------------------------------
st.subheader("Input Features")
col1, col2 = st.columns(2)

with col1:
    src_bytes = st.number_input("src_bytes", value=form_data["src_bytes"])
    dst_bytes = st.number_input("dst_bytes", value=form_data["dst_bytes"])
    logged_in = st.selectbox("logged_in", [0, 1], index=form_data["logged_in"])
    count = st.number_input("count", value=form_data["count"])
    serror_rate = st.number_input("serror_rate", min_value=0.0, max_value=1.0, value=form_data["serror_rate"])
    srv_serror_rate = st.number_input("srv_serror_rate", min_value=0.0, max_value=1.0, value=form_data["srv_serror_rate"])
    same_srv_rate = st.number_input("same_srv_rate", min_value=0.0, max_value=1.0, value=form_data["same_srv_rate"])
    diff_srv_rate = st.number_input("diff_srv_rate", min_value=0.0, max_value=1.0, value=form_data["diff_srv_rate"])

with col2:
    dst_host_count = st.number_input("dst_host_count", value=form_data["dst_host_count"])
    dst_host_srv_count = st.number_input("dst_host_srv_count", value=form_data["dst_host_srv_count"])
    dst_host_same_srv_rate = st.number_input("dst_host_same_srv_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_same_srv_rate"])
    dst_host_diff_srv_rate = st.number_input("dst_host_diff_srv_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_diff_srv_rate"])
    dst_host_same_src_port_rate = st.number_input("dst_host_same_src_port_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_same_src_port_rate"])
    dst_host_srv_diff_host_rate = st.number_input("dst_host_srv_diff_host_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_srv_diff_host_rate"])
    dst_host_serror_rate = st.number_input("dst_host_serror_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_serror_rate"])
    dst_host_srv_serror_rate = st.number_input("dst_host_srv_serror_rate", min_value=0.0, max_value=1.0, value=form_data["dst_host_srv_serror_rate"])

service = st.selectbox("Service", ["http", "smtp", "ftp", "dns", "ssh"], index=["http","smtp","ftp","dns","ssh"].index(form_data["service"]))
flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "SH"], index=["SF","S0","REJ","RSTR","SH"].index(form_data["flag"]))

# ------------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------------
predict_pressed = st.button(" Predict", key="predict", type="primary")

# ------------------------------------------------------------------
# PREPROCESS AND PREDICT
# ------------------------------------------------------------------
def preprocess_input():
    input_dict = {
        "src_bytes": src_bytes, "dst_bytes": dst_bytes, "logged_in": logged_in,
        "count": count, "serror_rate": serror_rate, "srv_serror_rate": srv_serror_rate,
        "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
        "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate, "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
        "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": dst_host_serror_rate, "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
        "service": service, "flag": flag
    }
    X = pd.DataFrame([input_dict])
    X = pd.get_dummies(X, columns=["service", "flag"], drop_first=True)
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[model_columns]
    X = scaler.transform(X)
    return X

# ------------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------------


if predict_pressed:
    X_processed = preprocess_input()
    pred = model.predict(X_processed)[0][0]

    # Determine label & confidence
    if pred < 0.5:
        label = "Normal"
        confidence = (1 - pred) * 100
        color = "#0e7b29"
        emoji = "✔"
    else:
        label = "Attack"
        confidence = pred * 100
        color = "#b00020"
        emoji = "⚠️"

    # Prediction Card
    st.markdown(
        f"<div style='background-color:{color}; padding:18px; border-radius:12px; color:white; font-size:22px; font-weight:bold; text-align:center;'>"
        f"{emoji} Prediction: {label} (Confidence: {confidence:.1f}%)</div>",
        unsafe_allow_html=True
    )

    # Animated confidence bar
    progress_container = st.empty()
    for i in range(int(confidence)+1):
        progress_container.markdown(
            f"<div class='progress-bar-container'>"
            f"<div class='progress-bar-fill' style='width:{i}%; background-color:{color};'>{i:.0f}%</div></div>",
            unsafe_allow_html=True
        )
        time.sleep(0.01)

    # -----------------------
    # LIME EXPLAINER
    # -----------------------
    st.subheader("📝 Explainable AI: Feature Impact (LIME)")

    # Convert X_processed back to original scaled values for LIME
    X_scaled = X_processed
    feature_names = model_columns
    class_names = ["Normal", "Attack"]

    explainer = LimeTabularExplainer(
        training_data=np.array(X_scaled),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=X_scaled[0],
        predict_fn=lambda x: np.hstack([1 - model.predict(x), model.predict(x)])
    )

    # Show top features as a table
    lime_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Contribution"])
    st.dataframe(lime_df)
