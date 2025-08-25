import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conlist

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"


class PredictInput(BaseModel):
    gender: str = Field(..., description="Male or Female")
    age: float
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


class PredictResponse(BaseModel):
    risk_score: float
    risk_label: str
    top_factors: List[str]
    echo_features: dict


class LocalExplainInput(PredictInput):
    top_k: int = 5


app = FastAPI(title="Diabetes Risk Model Service", version="0.1.0")

# Serve artifacts for clinician visuals
if ARTIFACTS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ARTIFACTS_DIR)), name="assets")


def _load_model_and_threshold():
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise RuntimeError("Model artifacts not found. Please run training first.")
    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        tconf = json.load(f)
    return model, float(tconf.get("threshold", 0.5))


MODEL, THRESHOLD = _load_model_and_threshold()


@app.get("/health")
def health():
    return {"status": "ok"}


def _label_from_score(score: float, threshold: float) -> str:
    if score >= threshold + 0.15:
        return "high"
    if score >= threshold:
        return "medium"
    return "low"


def _extract_top_factors(x: pd.DataFrame, k: int = 3) -> List[str]:
    # simple heuristic using absolute z-scores for numeric columns
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    vals = []
    for c in num_cols:
        v = float(x.iloc[0][c])
        vals.append((c, abs(v)))
    ranked = sorted(vals, key=lambda t: t[1], reverse=True)
    return [c for c, _ in ranked[:k]]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictInput):
    x = pd.DataFrame([{**payload.dict()}])
    try:
        proba = float(MODEL.predict_proba(x)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    label = _label_from_score(proba, THRESHOLD)
    top = _extract_top_factors(x, k=3)
    return PredictResponse(risk_score=proba, risk_label=label, top_factors=top, echo_features=payload.dict())


@app.get("/explain/global")
def explain_global():
    assets = {
        "shap_summary": "/assets/shap_summary.png",
        "pdp_glucose": "/assets/pdp_glucose.png",
        "pdp_bmi": "/assets/pdp_bmi.png",
        "metrics": "/assets/metrics.csv",
    }
    return assets


@app.post("/explain/local")
def explain_local(payload: LocalExplainInput):
    x = pd.DataFrame([{**payload.dict()}])
    try:
        proba = float(MODEL.predict_proba(x)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    # placeholder: reuse heuristic ranking; could be upgraded to true SHAP per-request if needed
    k = int(payload.top_k)
    top = _extract_top_factors(x, k=k)
    return {
        "risk_score": proba,
        "top_contributions": top,
    }


