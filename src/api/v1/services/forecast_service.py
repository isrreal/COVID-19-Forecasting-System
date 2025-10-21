import os
import mlflow
import joblib
import torch
import pandas as pd
import numpy as np
from datetime import date, timedelta
from mlflow.tracking import MlflowClient
from src.models.casos_covid import CasoCovid
from sqlalchemy import select, desc

from database import sync_engine

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
client = MlflowClient()

model_cache = {}

def _get_best_run_id_for_state(state_code: str) -> tuple[str, int] | None:
    """
    Retorna a run com o menor RMSE de VALIDAÇÃO para um estado.
    CORRIGIDO: Agora busca por val_rmse ao invés de train_rmse
    """
    experiment_name = f"Covid Forecasting Comparison - {state_code}"
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experimento '{experiment_name}' não encontrado.")
            return None

        runs_df = mlflow.search_runs(
            experiment_ids = [experiment.experiment_id],
            order_by = ["metrics.val_rmse ASC"],  
            max_results = 1
        )
        
        if runs_df.empty:
            print(f"Nenhuma run encontrada para o estado {state_code}.")
            return None
        
        best_run = runs_df.iloc[0]
        run_id = best_run["run_id"]
        
        if "params.sequence_length" in best_run:
            seq_length = int(best_run["params.sequence_length"])
        else:
            print(f"AVISO: 'sequence_length' não encontrado nos parâmetros da run {run_id}. Usando valor padrão 14.")
            seq_length = 14

        print(f"Melhor Run ID para {state_code}: {run_id} (Val RMSE: {best_run.get('metrics.val_rmse', 'N/A'):.2f})")
        return run_id, seq_length
    except Exception as e:
        print(f"Erro ao buscar melhor run para {state_code}: {e}")
        return None

def _load_model_from_mlflow(state_code: str):
    """
    Carrega o modelo PyTorch e o scaler a partir do MLflow, usando um cache.
    """
    if state_code in model_cache:
        print(f"Modelo para {state_code} recuperado do cache.")
        return model_cache[state_code]

    result = _get_best_run_id_for_state(state_code)
    if not result:
        return None
    
    run_id, seq_length = result

    try:
        local_dir = client.download_artifacts(run_id, "preprocessor", "/tmp")
        scaler_files = [f for f in os.listdir(local_dir) if f.endswith(".pkl") or f.endswith(".gz")]
        if not scaler_files:
            raise FileNotFoundError(f"Arquivo do scaler não encontrado no diretório: {local_dir}")
        scaler_path = os.path.join(local_dir, scaler_files[0])
        scaler = joblib.load(scaler_path)
        print(f"Scaler carregado com sucesso. Range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")
    except Exception as e:
        print(f"Erro ao carregar scaler da run {run_id}: {e}")
        return None

    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        print(f"Modelo PyTorch carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar modelo PyTorch da run {run_id}: {e}")
        return None

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
    Realiza uma única previsão para uma sequência de dados.
    CORRIGIDO: Agora aplica inverse_transform corretamente
    """
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return None

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    run_id = artifacts["run_id"]
    seq_length = artifacts["seq_length"]
    
    if len(sequence) != seq_length:
        raise ValueError(f"Tamanho da sequência ({len(sequence)}) diferente do esperado ({seq_length}).")

    sequence_np = np.array(sequence, dtype=np.float32).reshape(-1, 1)
    
    print(f"\nDEBUG - Sequência de entrada:")
    print(f"  Min: {sequence_np.min():.2f}, Max: {sequence_np.max():.2f}, Mean: {sequence_np.mean():.2f}")
    
    data_scaled = scaler.transform(sequence_np)
    print(f"DEBUG - Após scaling:")
    print(f"  Min: {data_scaled.min():.6f}, Max: {data_scaled.max():.6f}, Mean: {data_scaled.mean():.6f}")
    
    input_tensor = torch.from_numpy(data_scaled).float().view(1, seq_length, 1)

    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()
    
    print(f"DEBUG - Predição escalada: {prediction_scaled[0][0]:.6f}")
    
    prediction_inversed = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    predicted_cases = float(prediction_inversed[0, 0])
    
    print(f"DEBUG - Predição real (após inverse): {predicted_cases:.2f}")

    return {
        "state": state_code,
        "model_run_id": run_id,
        "prediction": round(max(0, predicted_cases), 2)
    }

def get_forecast_for_state(state_code: str, days: int) -> dict:
    """
    Gera uma previsão multi-step para N dias futuros, usando os dados mais recentes do banco.
    CORRIGIDO: Agora usa valores escalados na sequência interna
    """
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return None
    
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    run_id = artifacts["run_id"]
    seq_length = artifacts["seq_length"]

    with sync_engine.connect() as session:    
        query = (
            select(CasoCovid.datetime, CasoCovid.new_confirmed)
            .where(CasoCovid.state == state_code)
            .order_by(desc(CasoCovid.datetime))
            .limit(seq_length)
        )
        result = session.execute(query).all()

    if len(result) < seq_length:
        raise ValueError(f"Dados insuficientes para {state_code}: {len(result)} encontrados, {seq_length} necessários.")

    last_known_date = result[0][0]
    print(f"\n{'='*60}")
    print(f"FORECAST para {state_code} - {days} dias")
    print(f"Última data registrada: {last_known_date}")
    print(f"{'='*60}")

    latest_data_raw = np.array([row[1] for row in reversed(result)], dtype=np.float32)
    
    print(f"\nDados iniciais (raw):")
    print(f"  Min: {latest_data_raw.min():.2f}, Max: {latest_data_raw.max():.2f}, Mean: {latest_data_raw.mean():.2f}")
    
    current_sequence_scaled = scaler.transform(latest_data_raw.reshape(-1, 1)).flatten()
    
    forecast_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for i in range(1, days + 1):
        future_date = last_known_date + timedelta(days=i)

        input_tensor = torch.FloatTensor(current_sequence_scaled).view(1, seq_length, 1).to(device)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()[0, 0]
        
        pred_real = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predicted_value = max(0, float(pred_real))

        forecast_list.append({
            "date": future_date.strftime("%Y-%m-%d"),  
            "predicted_value": round(predicted_value, 2)
        })

        print(f"  Dia {i} ({future_date.strftime('%Y-%m-%d')}): {predicted_value:.2f} casos")

        current_sequence_scaled = np.append(current_sequence_scaled[1:], pred_scaled)

    print(f"{'='*60}\n")

    return {
        "state": state_code,
        "model_run_id": run_id,
        "last_known_date": last_known_date.strftime("%Y-%m-%d"),
        "sequence_length": seq_length,
        "forecast": forecast_list
    }