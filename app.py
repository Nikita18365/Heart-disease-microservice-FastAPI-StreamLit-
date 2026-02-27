from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback

import xgboost as xgb
import cupy as cp

from feature_engineering import build_features, FEArtifacts

app = FastAPI(title = "Heart Disease Predictor", version = "1.0.0")

# Load model + artifacts once on startup
model = joblib.load("model.joblib")
booster = model.get_booster()
artifacts: FEArtifacts = joblib.load("artifacts.joblib")

THRESHOLD = 0.5


class Patient(BaseModel):
    Age: int = Field(..., ge = 0, le = 120)
    Sex: int = Field(..., ge = 0, le = 1)

    Chest_pain_type: int = Field(..., ge = 1, le = 4, alias = "Chest pain type")
    BP: int = Field(..., ge = 0, le = 300)
    Cholesterol: int = Field(..., ge = 0, le = 700)

    FBS_over_120: int = Field(..., ge = 0, le = 1, alias = "FBS over 120")
    EKG_results: int = Field(..., ge = 0, le = 2, alias = "EKG results")

    Max_HR: int = Field(..., ge = 0, le = 250, alias = "Max HR")
    Exercise_angina: int = Field(..., ge = 0, le = 1, alias = "Exercise angina")

    ST_depression: float = Field(..., ge = 0, le = 10, alias = "ST depression")
    Slope_of_ST: int = Field(..., ge = 0, le = 3, alias = "Slope of ST")

    Number_of_vessels_fluro: int = Field(..., ge = 0, le = 3, alias = "Number of vessels fluro")
    Thallium: int = Field(..., ge = 0, le = 7)


@app.get("/health")
def health():
    # Quick check that the service is alive (without calculating the features/model)
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    # Convert request -> DataFrame with original column names (aliases)
    row = patient.model_dump(by_alias = True)
    df = pd.DataFrame([row])

    # Feature engineering (same as training)
    X = build_features(df, artifacts)
    FEATURES = joblib.load("features.joblib")
    #X = X.reindex(columns = FEATURES, fill_value = 0)
    
    if "id" in X.columns:
        X = X.drop(columns = ["id"])
    bad = X.select_dtypes(include = ["object"]).columns.tolist()
    if bad:
        return {"error": f"Object columns found: {bad}"}

    # GPU inference
    X_gpu = cp.asarray(X.values, dtype = cp.float32)
    dmat = xgb.DMatrix(X_gpu, feature_names = FEATURES)
    proba = float(booster.predict(dmat)[0])
    pred = int(proba >= THRESHOLD)

    return {"proba": proba, "pred": pred}
