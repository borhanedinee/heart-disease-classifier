from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ── Load trained model ───────────────────────────────────────────
model = joblib.load("model.pkl")

# ── Define app ───────────────────────────────────────────────────
app = FastAPI(title="Heart Disease Classifier API")

# ── Define input schema ──────────────────────────────────────────
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# ── Health check ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Heart Disease Classifier API is running 🚀"}

# ── Prediction endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientData):
    features = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "result": "Disease likely 🔴" if prediction == 1 else "Healthy 🟢",
        "confidence": round(float(probability), 4)
    }