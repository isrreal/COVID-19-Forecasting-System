import os
import mlflow
import joblib
import torch
import numpy as np
from datetime import timedelta, datetime as dt
from mlflow.tracking import MlflowClient
from src.models.caso_dengue import CasoDengue
from sqlalchemy import select, func
from scipy.stats import norm

from database import sync_engine

# ==========================================================
# General Configuration and Model Loading
# ==========================================================

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
client = MlflowClient()

model_cache: dict[str, dict] = {}

_STATE_IBGE_CODE: dict[str, int] = {
    "RO": 11,
    "AC": 12,
    "AM": 13,
    "RR": 14,
    "PA": 15,
    "AP": 16,
    "TO": 17,
    "MA": 21,
    "PI": 22,
    "CE": 23,
    "RN": 24,
    "PB": 25,
    "PE": 26,
    "AL": 27,
    "SE": 28,
    "BA": 29,
    "MG": 31,
    "ES": 32,
    "RJ": 33,
    "SP": 35,
    "PR": 41,
    "SC": 42,
    "RS": 43,
    "MS": 50,
    "MT": 51,
    "GO": 52,
    "DF": 53,
}


def _get_best_run_id_for_state(state_code: str) -> tuple[str, int] | None:
    """Returns the run with the lowest validation RMSE for a given state."""
    experiment_name = f"Dengue Forecasting Comparison - {state_code}"
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return None

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_rmse ASC"],
            max_results=1,
        )

        if runs_df.empty:
            print(f"No runs found for state {state_code}.")
            return None

        best_run = runs_df.iloc[0]
        run_id = best_run["run_id"]

        if "params.sequence_length" in best_run:
            seq_length = int(best_run["params.sequence_length"])
        else:
            print("WARNING: 'sequence_length' not found. Using default 14.")
            seq_length = 14

        print(
            f"Best Run ID for {state_code}: {run_id} (Val RMSE: {best_run.get('metrics.val_rmse', 'N/A'):.2f})"
        )
        return run_id, seq_length

    except Exception as e:
        print(f"Error fetching best run for {state_code}: {e}")
        return None


def _load_model_from_mlflow(state_code: str):
    """Loads the PyTorch model and scaler from MLflow, using an in-memory cache."""
    if state_code in model_cache:
        print(f"Model for {state_code} retrieved from cache.")
        return model_cache[state_code]

    result = _get_best_run_id_for_state(state_code)
    if not result:
        return None

    run_id, seq_length = result

    try:
        local_dir = client.download_artifacts(run_id, "preprocessor", "/tmp")
        scaler_files = [
            f for f in os.listdir(local_dir) if f.endswith(".pkl") or f.endswith(".gz")
        ]
        if not scaler_files:
            raise FileNotFoundError(f"Scaler file not found in {local_dir}")
        scaler_path = os.path.join(local_dir, scaler_files[0])
        scaler = joblib.load(scaler_path)
        print(
            f"Scaler loaded. Range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]"
        )
    except Exception as e:
        print(f"Error loading scaler from run {run_id}: {e}")
        return None

    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Error loading PyTorch model from run {run_id}: {e}")
        return None

    model_cache[state_code] = {
        "model": model,
        "scaler": scaler,
        "run_id": run_id,
        "seq_length": seq_length,
    }

    print(f"Model for {state_code} loaded and saved to cache.")
    return model_cache[state_code]


# ==========================================================
# Single-step Prediction
# ==========================================================


def get_prediction_for_state(state_code: str, sequence: list) -> dict | None:
    """Performs a single-step prediction for a given input sequence."""
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    run_id = artifacts["run_id"]
    seq_length = artifacts["seq_length"]

    if len(sequence) != seq_length:
        raise ValueError(
            f"Sequence length ({len(sequence)}) does not match expected ({seq_length})."
        )

    sequence_np = np.array(sequence, dtype=np.float32).reshape(-1, 1)
    data_scaled = scaler.transform(sequence_np)
    input_tensor = torch.from_numpy(data_scaled).float().unsqueeze(0)

    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()

    prediction_inversed = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    predicted_cases = float(prediction_inversed[0, 0])

    return {
        "state": state_code,
        "model_run_id": run_id,
        "prediction": round(max(0, predicted_cases), 2),
    }


def _get_initial_sequence(
    session, state_code: str, seq_length: int, municipality_code: int | None = None
) -> tuple[list[float], dt] | None:
    """Fetches the most recent daily case counts to seed the autoregressive loop."""
    ibge_code = _STATE_IBGE_CODE.get(state_code.upper())
    if ibge_code is None:
        return None

    query = select(
        CasoDengue.notification_date,
        func.count().label("daily_cases"),
    ).where(CasoDengue.state_ibge_code == ibge_code)

    if municipality_code is not None:
        query = query.where(CasoDengue.municipality_ibge_code == municipality_code)

    query = (
        query.group_by(CasoDengue.notification_date)
        .order_by(CasoDengue.notification_date.desc())
        .limit(seq_length)
    )

    result = session.execute(query).all()

    if not result or len(result) < seq_length:
        return None

    result.reverse()

    last_date = dt.combine(result[-1].notification_date, dt.min.time())
    initial_sequence_values = [float(r.daily_cases) for r in result]

    return initial_sequence_values, last_date


def _generate_autoregressive_forecast(
    model, scaler, initial_sequence: list, seq_length: int, days: int, last_date: dt
) -> list[dict]:
    """Runs the autoregressive forecast loop for the given number of days."""
    current_sequence_scaled = scaler.transform(
        np.array(initial_sequence, dtype=np.float32).reshape(-1, 1)
    )

    forecast = []
    for i in range(days):
        with torch.no_grad():
            input_tensor = (
                torch.from_numpy(current_sequence_scaled).float().unsqueeze(0)
            )
            pred_scaled = model(input_tensor).cpu().numpy()

        pred_real = scaler.inverse_transform(pred_scaled)[0][0]
        pred_real = max(0, pred_real)

        forecast.append(
            {
                "date": (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "predicted_value": round(float(pred_real), 2),
            }
        )

        current_sequence_scaled = np.vstack(
            (current_sequence_scaled[1:], pred_scaled.reshape(1, 1))
        )

    return forecast


def get_forecast_for_entire_state(state_code: str, days: int) -> dict:
    """Returns a multi-step forecast for the entire aggregated state."""
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Model not found for {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    with sync_engine.connect() as session:
        sequence_data = _get_initial_sequence(
            session, state_code, seq_length, municipality_code=None
        )

        if not sequence_data:
            return {"error": f"Insufficient data for {state_code}."}

        initial_sequence, last_date = sequence_data

        forecast = _generate_autoregressive_forecast(
            model, scaler, initial_sequence, seq_length, days, last_date
        )

    return {"state": state_code, "model_run_id": run_id, "forecast": forecast}


def get_forecast_for_municipality(
    state_code: str, municipality_code: int, days: int
) -> dict:
    """Returns a multi-step forecast for a specific municipality within a state."""
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Model not found for {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    with sync_engine.connect() as session:
        sequence_data = _get_initial_sequence(
            session, state_code, seq_length, municipality_code=municipality_code
        )

        if not sequence_data:
            return {
                "error": f"Insufficient data for municipality {municipality_code}, {state_code}."
            }

        initial_sequence, last_date = sequence_data

        forecast = _generate_autoregressive_forecast(
            model, scaler, initial_sequence, seq_length, days, last_date
        )

    return {
        "state": state_code,
        "municipality_code": municipality_code,
        "model_run_id": run_id,
        "forecast": forecast,
    }


def get_forecast_for_state(state_code: str, days: int) -> dict:
    """Returns multi-step forecasts for all municipalities within a state."""
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Model not found for {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    ibge_code = _STATE_IBGE_CODE.get(state_code.upper())
    if ibge_code is None:
        return {"error": f"Unknown state abbreviation: {state_code}"}

    forecasts_by_municipality = {}
    with sync_engine.connect() as session:
        municipalities_query = (
            select(CasoDengue.municipality_ibge_code)
            .where(CasoDengue.state_ibge_code == ibge_code)
            .distinct()
        )
        municipalities = [
            r.municipality_ibge_code
            for r in session.execute(municipalities_query).all()
        ]

        if not municipalities:
            return {"error": f"No municipalities found for {state_code}"}

        for municipality_code in municipalities:
            sequence_data = _get_initial_sequence(
                session, state_code, seq_length, municipality_code=municipality_code
            )

            if not sequence_data:
                print(
                    f"WARNING: Insufficient data for municipality {municipality_code} (required: {seq_length})"
                )
                continue

            initial_sequence, last_date = sequence_data

            municipality_forecast = _generate_autoregressive_forecast(
                model, scaler, initial_sequence, seq_length, days, last_date
            )

            forecasts_by_municipality[str(municipality_code)] = municipality_forecast

    return {
        "state": state_code,
        "model_run_id": run_id,
        "forecasts": forecasts_by_municipality,
    }


def get_forecast_with_confidence(
    state_code: str, days: int, confidence: float = 0.95
) -> dict:
    """Returns a multi-step forecast with confidence intervals for the aggregated state."""
    base_forecast = get_forecast_for_entire_state(state_code, days)
    if "forecast" not in base_forecast:
        return {"error": f"Failed to generate forecast for {state_code}"}

    predictions = np.array(
        [item["predicted_value"] for item in base_forecast["forecast"]]
    )

    std_dev = np.maximum(predictions * 0.05, 1.0)
    z_score = norm.ppf(1 - (1 - confidence) / 2)

    forecast_with_ci = []
    for i, item in enumerate(base_forecast["forecast"]):
        forecast_with_ci.append(
            {
                "date": item["date"],
                "predicted_mean": round(float(predictions[i]), 2),
                "lower_bound": round(max(0, predictions[i] - z_score * std_dev[i]), 2),
                "upper_bound": round(predictions[i] + z_score * std_dev[i], 2),
            }
        )

    return {
        "state": state_code,
        "model_run_id": base_forecast["model_run_id"],
        "confidence_level": confidence,
        "forecast_with_confidence": forecast_with_ci,
    }
