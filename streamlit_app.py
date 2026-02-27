from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import cupy as cp
import xgboost as xgb

import streamlit as st
import requests

# For Local prediction
from feature_engineering import build_features 

# Page config
st.set_page_config(page_title = "Heart Disease Risk Predictor",
                   page_icon  = "🫀",
                   layout     = "wide",
                  )

# Styling
st.markdown("""
            <style>
            .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
            .stMetric { background: #0b1220; padding: 14px; border-radius: 14px; }
            div[data-testid="stMetricValue"] { font-size: 26px; }
            </style>
            """,
            unsafe_allow_html = True, #!!!
           )

BASE_DIR = Path(__file__).resolve().parent

# Caching model/artifacts
@st.cache_resource
def load_local_assets():
    model     = joblib.load(BASE_DIR / "model.joblib")
    artifacts = joblib.load(BASE_DIR / "artifacts.joblib")
    # This is optional: a joblib file for the correct feature order
    features_path = BASE_DIR / "features.joblib"
    if features_path.exists():
        features = joblib.load(features_path)
    else:
        features = getattr(artifacts, "features", None)
    return model, artifacts, features

def make_payload(Age, Sex, ChestPain, BP, Chol, FBS, EKG, MaxHR, ExAng, STdep, SlopeST, Vessels, Thallium):
    # Important: the keys must match the names in feature_engineering
    return {"Age":                     int(Age),
            "Sex":                     int(Sex),
            "Chest pain type":         int(ChestPain),
            "BP":                      int(BP),
            "Cholesterol":             int(Chol),
            "FBS over 120":            int(FBS),
            "EKG results":             int(EKG),
            "Max HR":                  int(MaxHR),
            "Exercise angina":         int(ExAng),
            "ST depression":           float(STdep),
            "Slope of ST":             int(SlopeST),
            "Number of vessels fluro": int(Vessels),
            "Thallium":                int(Thallium),
           }

def predict_local(payload: dict):
    model, artifacts, features = load_local_assets()
    # GPU inference don't support via DMatrix, only CPU
    model.set_params(device = "cpu")
    df = pd.DataFrame([payload])
    X = build_features(df, artifacts)

    # Protection against the parasitic column "id"
    if "id" in X.columns:
        X = X.drop(columns = ["id"])

    # Guarantee the order of the columns
    if features is not None:
        X = X.reindex(columns = features, fill_value = 0)

    proba = float(model.predict_proba(X)[:, 1][0])
    return proba

def predict_via_api(payload: dict, api_url: str):
    r = requests.post(api_url.rstrip("/") + "/predict", json = payload, timeout = 10)
    r.raise_for_status()
    data = r.json()
    return float(data["proba"])

# Conf of prediction (risk label)
def risk_label(p: float):
    if p < 0.33:
        return "Low risk", "✅"
    elif p < 0.66:
        return "Medium риск", "⚠️"
    return "High risk", "🚨"

# Header
left, right = st.columns([0.7, 0.3], 
                         vertical_alignment = "center")
with left:
    st.title("🫀 Heart Disease Risk Predictor")
    st.caption("An interactive interface for assessing the risk of heart disease based on a boosting model.")
with right:
    st.info("⚙️ Select the inference mode on the left (locally / via FastAPI).")
st.divider()

# Sidebar settings
st.sidebar.header("Settings")
mode = st.sidebar.radio("Prediction mode", ["Local", "FastAPI"], index = 0)
api_url = None
if mode == "FastAPI":
    api_url = st.sidebar.text_input("API URL", value = "http://127.0.0.1:8000")
    st.sidebar.caption("Make sure that FastAPI is running and available at the address above")
threshold = st.sidebar.slider("Class threshold (pred = 1 if proba ≥ threshold)", 0.1, 0.9, 0.5, 0.01)
st.sidebar.caption("You can change the threshold to match the FP/FN balance.")
st.sidebar.divider()
if st.sidebar.button("🎲 Fill in the patient example"):
    st.session_state["example"] = True

# Input form
col1, col2, col3 = st.columns([0.34, 0.33, 0.33])
example = st.session_state.get("example", False)
with col1:
    st.subheader("👤 Patient")
    Age = st.number_input("Age (years)", min_value = 0, max_value = 120, value = 58 if example else 50, step = 1)
    Sex = st.selectbox("Sex", options = [0, 1], index = 1 if example else 0, help = "0 = Female, 1 = Male")
    ChestPain = st.selectbox("Chest pain type", options = [1, 2, 3, 4], index = 3 if example else 0,
                             help = "1 Typical angina, 2 Atypical, 3 Non-anginal, 4 Asymptomatic")
    FBS = st.selectbox("FBS over 120", options = [0, 1], index = 0 if example else 0, help = "1 = True, 0 = False")
with col2:
    st.subheader("🩺 Measurements")
    BP = st.number_input("BP (mm Hg)", min_value = 0, max_value = 300, value = 152 if example else 130, step = 1)
    Chol = st.number_input("Cholesterol (mg/dL)", min_value = 0, max_value = 700, value = 239 if example else 200, step = 1)
    MaxHR = st.number_input("Max HR", min_value = 0, max_value = 250, value = 158 if example else 150, step = 1)
    STdep = st.number_input("ST depression", min_value = 0.0, max_value = 10.0, value = 3.6 if example else 1.0, step = 0.1)
with col3:
    st.subheader("📈 Tests / Categories")
    EKG = st.selectbox("EKG results", options = [0, 1, 2], index = 0 if example else 0)
    ExAng = st.selectbox("Exercise angina", options = [0, 1], index = 1 if example else 0)
    SlopeST = st.selectbox("Slope of ST", options = [0, 1, 2, 3], index = 2 if example else 0)
    Vessels = st.selectbox("Number of vessels fluro", options = [0, 1, 2, 3], index = 2 if example else 0)
    Thallium = st.selectbox("Thallium", options = list(range(0, 8)), index = 7 if example else 0)
st.session_state["example"] = False

payload = make_payload(Age, Sex, ChestPain, BP, Chol, FBS, EKG, MaxHR, ExAng, STdep, SlopeST, Vessels, Thallium)
with st.expander("🔎 View the JSON that is sent to the model"):
    st.json(payload)

# Predict button
st.divider()
btn_col1, btn_col2 = st.columns([0.25, 0.75])
with btn_col1:
    run = st.button("🚀 Predict", use_container_width = True)
with btn_col2:
    st.caption("If FastAPI mode is selected, make sure that the service is running: `uvicorn app:app --port 8000`")
if run:
    try:
        with st.spinner("I consider it a risk..."):
            if mode == "Local":
                proba = predict_local(payload)
            else:
                proba = predict_via_api(payload, api_url)
        pred = int(proba >= threshold)
        label, icon = risk_label(proba)
        # Results
        # Results (styled card)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
        <div style="
            background-color: #e9f7ef;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #b7e4c7;
            text-align: center;
            ">
            <p style="color: black; font-size: 14px; margin: 0;">
                <b>Probability (proba)</b>
            </p>
            <p style="color: black; font-size: 28px; margin: 4px 0;">
                {proba:.4f}
            </p>
        </div>
        """,
            unsafe_allow_html=True
                       )
        with c2:
            st.markdown(f"""
        <div style="
            background-color: #e9f7ef;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #b7e4c7;
            text-align: center;
            ">
            <p style="color: black; font-size: 14px; margin: 0;">
                <b>Class (pred)</b>
            </p>
            <p style="color: black; font-size: 26px; margin: 4px 0;">
                {pred} <span style="font-size:14px;">(threshold={threshold:.2f})</span>
            </p>
        </div>
        """,
            unsafe_allow_html=True
                       )
        with c3:
            st.markdown(f"""
        <div style="
            background-color: #e9f7ef;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #b7e4c7;
            text-align: center;
            ">
            <p style="color: black; font-size: 14px; margin: 0;">
                <b>Interpretation of risk</b>
            </p>
            <p style="color: black; font-size: 22px; margin: 4px 0;">
                {icon} {label}
            </p>
        </div>
        """,
            unsafe_allow_html=True
                       )
        # Gauge-like bar
        st.progress(min(max(proba, 0.0), 1.0))
        st.markdown("### 🧠 Explanation")
        if pred == 1:
            st.warning("The model considers the risk to be increased (pred = 1). This is not a diagnosis — you need a doctor.")
        else:
            st.success("The model considers the risk to be low (pred = 0). This is not a diagnosis — if you have symptoms, consult a doctor.")
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("If this is a local mode, make sure that model.joblib/artifacts.joblib are located next to streamlit_app.py.")
