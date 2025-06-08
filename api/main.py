# File: api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load model at startup
model = joblib.load("api/model/heart_disease_model.pkl")

# ✅ 8 Features used in training
class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: float = Field(..., ge=80, le=200, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=600, description="Cholesterol in mg/dl")
    thalch: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression")
    dataset: float = Field(..., description="Dataset ID")
    id: float = Field(..., description="Patient ID")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: PatientData):
    try:
        features = np.array([
            data.age,
            data.cp,
            data.trestbps,
            data.chol,
            data.thalch,
            data.oldpeak,
            data.dataset,
            data.id
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(proba) if proba is not None else None,
            "message": "✅ Prediction successful"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
