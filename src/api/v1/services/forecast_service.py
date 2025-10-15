import os
import mlflow
import joblib
import torch
import pandas as pd
import numpy as np
from datetime import date, timedelta
from mlflow.tracking import MlflowClient

from database import sync_engine 


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
client = MlflowClient()

model_cache = {}

def _get_best_run_id_for_state(state_code: str) -> tuple[str, int] | None:
    """
    Busca no MLflow e retorna o ID da melhor run para um estado específico.
    A "melhor" run é definida como aquela com o menor `train_rmse`.
    """
    experiment_name = f"Covid Forecasting - {state_code}"
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experimento '{experiment_name}' não encontrado.")
            return None

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.train_rmse ASC"],
            max_results=1
        )
        if runs_df.empty:
            print(f"Nenhuma run encontrada para o estado {state_code}.")
            return None
        
        best_run = runs_df.iloc[0]
        run_id = best_run["run_id"]
        seq_length = int(best_run["params.sequence_length"])
        print(f"Melhor Run ID para {state_code}: {run_id} (RMSE: {best_run['metrics.train_rmse']:.2f})")
        return run_id, seq_length
    except Exception as e:
        print(f"Erro ao buscar melhor run para {state_code}: {e}")
        return None

def _load_model_from_mlflow(state_code: str):
    """
    Carrega o modelo e o scaler do MLflow para a memória (e para o cache).
    """
    if state_code in model_cache:
        print(f"Modelo para {state_code} encontrado no cache.")
        return model_cache[state_code]

    print(f"Modelo para {state_code} não encontrado no cache. Carregando do MLflow...")
    result = _get_best_run_id_for_state(state_code)
    if not result:
        return None
    
    run_id, seq_length = result

    local_scaler_dir = client.download_artifacts(run_id, "scaler", "/tmp")
    scaler_path = os.path.join(local_scaler_dir, f"scaler_{run_id}.gz")
    scaler = joblib.load(scaler_path)

    model_uri = f"runs:/{run_id}/pytorch-model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    model_cache[state_code] = {
        "model": model,
        "scaler": scaler,
        "run_id": run_id,
        "seq_length": seq_length
    }
    print(f"Modelo para {state_code} carregado e salvo no cache.")
    return model_cache[state_code]


def get_prediction_for_state(state_code: str, sequence: list) -> dict:
    """
    Carrega o modelo para um estado e faz uma previsão one-step-ahead.
    Esta função agora usa a LÓGICA REAL.
    """
    loaded_artifacts = _load_model_from_mlflow(state_code)
    if not loaded_artifacts:
        return None

    model = loaded_artifacts["model"]
    scaler = loaded_artifacts["scaler"]
    run_id = loaded_artifacts["run_id"]
    seq_length = loaded_artifacts["seq_length"]
    
    if len(sequence) != seq_length:
        raise ValueError(f"O tamanho da sequência de entrada ({len(sequence)}) é diferente do esperado pelo modelo ({seq_length}).")

    sequence_np = np.array(sequence).reshape(-1, 1)
    data_scaled = scaler.transform(sequence_np)
    input_tensor = torch.from_numpy(data_scaled).float().view(1, seq_length, 1)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    prediction_inversed = scaler.inverse_transform(prediction_scaled.numpy())
    predicted_cases = prediction_inversed[0][0]

    return {
        "state": state_code,
        "model_run_id": run_id,
        "prediction": round(float(predicted_cases), 2)
    }


def get_forecast_for_state(state_code: str, days: int) -> dict:
    """
    Busca os dados mais recentes e gera uma previsão para múltiplos dias de forma recursiva.
    Esta função agora usa a LÓGICA REAL.
    """
    loaded_artifacts = _load_model_from_mlflow(state_code)
    if not loaded_artifacts:
        return None
    
    run_id = loaded_artifacts["run_id"]
    seq_length = loaded_artifacts["seq_length"]

    print(f"Buscando os últimos {seq_length} dias de dados do banco para {state_code}...")
    query = f"SELECT new_confirmed FROM casos_covid WHERE state='{state_code}' ORDER BY datetime DESC LIMIT {seq_length}"
    latest_data_df = pd.read_sql(query, sync_engine)
    
    if len(latest_data_df) < seq_length:
        raise ValueError(f"Dados insuficientes no banco para {state_code}. Encontrados: {len(latest_data_df)}, Necessários: {seq_length}.")

    current_sequence = latest_data_df['new_confirmed'].values[::-1].tolist()
    
    forecast_list = []
    today = date.today()

    for i in range(1, days + 1):
        result = get_prediction_for_state(state_code, current_sequence)
        predicted_value = result["prediction"]

        future_date = today + timedelta(days=i)
        forecast_list.append({"date": future_date, "predicted_value": predicted_value})

        current_sequence.pop(0)
        current_sequence.append(predicted_value)

    return {
        "state": state_code,
        "model_run_id": run_id,
        "forecast": forecast_list
    }