from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import logging
import json
import time
from datetime import datetime

app = FastAPI(title="Churn Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.log"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("churn_api")


class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    start_time = time.time()

    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    latency_ms = round((time.time() - start_time) * 1000, 2)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_dict,
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "latency_ms": latency_ms
    }
    # Log as real JSON (one line per prediction) instead of Python dict repr
    logger.info(json.dumps(log_entry))

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "result": "Will Churn" if prediction == 1 else "Will Not Churn"
    }

@app.get("/health")
def health():
    history = load_prediction_history()
    drift_info = check_drift()

    avg_latency = None
    if not history.empty and "latency_ms" in history.columns:
        avg_latency = round(float(history["latency_ms"].mean()), 2)

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_predictions_served": len(history),
        "average_latency_ms": avg_latency,
        "drift_check": drift_info
    }

def load_prediction_history():
    """Read all logged predictions back into a DataFrame for analysis."""
    if not LOG_FILE.exists():
        return pd.DataFrame()

    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    return df


import numpy as np

DATA_PATH = BASE_DIR / "data" / "Telco_Churn.csv"


def check_drift():
    """
    Compare live prediction request data against original training data
    for a few key numeric features. Returns a simple report per feature:
    the training mean/std vs the live mean/std, and whether the live mean
    falls outside a reasonable range of the training distribution.
    """
    history = load_prediction_history()

    if history.empty or len(history) < 3:
        return {"status": "insufficient_data", "message": "Not enough prediction history yet to assess drift."}

    # Load original training data for comparison
    train_df = pd.read_csv(DATA_PATH)
    train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
    train_df.dropna(inplace=True)

    features_to_check = ["input.tenure", "input.MonthlyCharges", "input.TotalCharges"]
    train_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    drift_report = {}

    for live_col, train_col in zip(features_to_check, train_columns):
        if live_col not in history.columns:
            continue

        live_mean = history[live_col].mean()
        live_std = history[live_col].std()
        train_mean = train_df[train_col].mean()
        train_std = train_df[train_col].std()

        # Simple rule: flag drift if live mean is more than 2 training-std-devs away from training mean
        threshold = 2 * train_std
        is_drifted = abs(live_mean - train_mean) > threshold

        drift_report[train_col] = {
            "training_mean": round(float(train_mean), 2),
            "live_mean": round(float(live_mean), 2) if not np.isnan(live_mean) else None,
            "difference": round(float(live_mean - train_mean), 2) if not np.isnan(live_mean) else None,
            "drift_detected": bool(is_drifted) if not np.isnan(live_mean) else False
        }

    return {
        "status": "ok",
        "total_predictions_logged": len(history),
        "drift_report": drift_report
    }