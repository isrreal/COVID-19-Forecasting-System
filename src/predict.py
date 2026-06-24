import os
from pathlib import Path

import joblib
import mlflow
import torch
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import func, select

from database import sync_engine
from src.models.caso_dengue import CasoDengue
from src.train import _STATE_IBGE_CODE

BEST_RUN_ID = "REPLACE_WITH_YOUR_BEST_RUN_ID"
SEQ_LENGTH = 14
STATE = "CE"


def predict_next_day(run_id: str, state: str = STATE) -> float | None:
    """Loads the best MLflow model and predicts daily dengue cases for the next day."""
    if run_id == "REPLACE_WITH_YOUR_BEST_RUN_ID":
        raise ValueError("Update BEST_RUN_ID with a valid MLflow run ID.")

    ibge_code = _STATE_IBGE_CODE.get(state.upper())
    if ibge_code is None:
        print(f"Unknown state abbreviation: {state}")
        return None

    print(f"Starting prediction using run ID: {run_id}")

    try:
        client = MlflowClient()
        local_scaler_dir = client.download_artifacts(run_id, "preprocessor", "/tmp")
        scaler_path = Path(local_scaler_dir) / "scaler.pkl"
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    print(f"Fetching last {SEQ_LENGTH} days of dengue data for state {state}...")
    with sync_engine.connect() as conn:
        query = (
            select(
                CasoDengue.notification_date,
                func.count().label("daily_cases"),
            )
            .where(CasoDengue.state_ibge_code == ibge_code)
            .group_by(CasoDengue.notification_date)
            .order_by(CasoDengue.notification_date.desc())
            .limit(SEQ_LENGTH)
        )
        df = pd.DataFrame(
            conn.execute(query).fetchall(), columns=["date", "daily_cases"]
        )

    if len(df) < SEQ_LENGTH:
        print(f"Insufficient data: found {len(df)} records, need {SEQ_LENGTH}.")
        return None

    sequence = df["daily_cases"].values[::-1].astype(float).reshape(-1, 1)
    data_scaled = scaler.transform(sequence)
    input_tensor = torch.from_numpy(data_scaled).float().unsqueeze(0)

    print("Running prediction...")
    with torch.inference_mode():
        prediction_scaled = model(input_tensor).cpu().numpy()

    predicted_cases = float(
        scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
    )
    predicted_cases = max(0, predicted_cases)

    print("-" * 50)
    print(f"Predicted dengue cases for the next day ({state}): {predicted_cases:.2f}")
    print("-" * 50)

    return predicted_cases


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    )
    predict_next_day(run_id=BEST_RUN_ID, state=STATE)
