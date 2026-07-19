from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model and scaler (use relative paths if possible)
model = joblib.load("results/Random_Forest.pkl")
scaler = joblib.load("results/scaler.pkl")

# Input schema with all features
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "endpoint": "send POST request to /predict"
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestClassifier",
        "scaler_loaded": True
    }

@app.post("/predict")
def predict(trans: Transaction):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([trans.dict()])

        # Scale features
        scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed: {str(e)}"
        )
