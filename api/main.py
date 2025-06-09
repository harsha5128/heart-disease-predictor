from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# === Load model on startup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_disease_model.pkl")
model = joblib.load(MODEL_PATH)

# === Input Schema (Matching Gold Layer Columns used in model training) ===
class PatientData(BaseModel):
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    thalch: float = Field(..., ge=60, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes, 0 = no)")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type")
    dataset: float = Field(..., description="Dataset identifier")
    id: float = Field(..., description="Patient identifier")
    sex: float = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")

# === Root Endpoint ===
@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}

# === Health Check ===
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# === Prediction Endpoint ===
@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Arrange features in the exact order as model was trained on
        features = np.array([
            data.oldpeak,
            data.thalch,
            data.exang,
            data.age,
            data.ca,
            data.cp,
            data.dataset,
            data.id,
            data.sex
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
