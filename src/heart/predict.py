import joblib
import pandas as pd

MODEL_PATH = "models/heart_model.joblib"
_model = None


def predict_heart(patient: dict) -> dict:
    """Take one patient's values, return disease probability."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    proba = float(_model.predict_proba(pd.DataFrame([patient]))[0, 1])
    return {"probability": round(proba, 3), "prediction": int(proba >= 0.5)}