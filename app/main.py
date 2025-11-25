from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="NSL-KDD Intrusion Detection API")

class PredictRequest(BaseModel):
    features: list

MODEL_PATH = os.path.join("models", "nsl_kdd_model.joblib")
model = None

# Try to load model at import time; if missing, API will return helpful error
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        # Keep model as None and return error on predict
        model = None
        print(f"Failed to load model: {e}")
else:
    print(f"Model not found at {MODEL_PATH}. Place your serialized model at this path.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found on server.")
    try:
        X = np.array(req.features).reshape(1, -1)
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).tolist()
        # Ensure JSON-serializable types
        return {"prediction": int(pred[0]), "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
