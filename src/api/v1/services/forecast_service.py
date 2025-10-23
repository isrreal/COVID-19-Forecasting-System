import os
import mlflow
import joblib
import torch
import numpy as np
from datetime import timedelta, datetime as dt
from mlflow.tracking import MlflowClient
from src.models.casos_covid import CasoCovid
from sqlalchemy import select, func
from scipy.stats import norm

from database import sync_engine

# ==========================================================
# Configurações gerais e Carregamento de Modelo
# ==========================================================

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
client = MlflowClient()

model_cache = {}

def _get_best_run_id_for_state(state_code: str) -> tuple[str, int] | None:
    """
    Retorna a run com o menor RMSE de VALIDAÇÃO para um estado.
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
            print(f"AVISO: 'sequence_length' não encontrado. Usando padrão 14.")
            seq_length = 14

        print(f"Melhor Run ID para {state_code}: {run_id} (Val RMSE: {best_run.get('metrics.val_rmse', 'N/A'):.2f})")
        return run_id, seq_length

    except Exception as e:
        print(f"Erro ao buscar melhor run para {state_code}: {e}")
        return None


def _load_model_from_mlflow(state_code: str):
    """
    Carrega o modelo PyTorch e o scaler a partir do MLflow, usando cache.
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
            raise FileNotFoundError(f"Arquivo do scaler não encontrado em {local_dir}")
        scaler_path = os.path.join(local_dir, scaler_files[0])
        scaler = joblib.load(scaler_path)
        print(f"Scaler carregado. Range: [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")
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


# ==========================================================
# Predição única
# ==========================================================

def get_prediction_for_state(state_code: str, sequence: list) -> dict:
    """
    Realiza uma única previsão para uma sequência de dados.
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

    sequence_np = np.array(sequence, dtype = np.float32).reshape(-1, 1)
    data_scaled = scaler.transform(sequence_np)
    input_tensor = torch.from_numpy(data_scaled).float().view(1, seq_length, 1)

    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()

    prediction_inversed = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    predicted_cases = float(prediction_inversed[0, 0])

    return {
        "state": state_code,
        "model_run_id": run_id,
        "prediction": round(max(0, predicted_cases), 2)
    }

def _get_initial_sequence(session, state_code: str, seq_length: int, city: str | None = None) -> tuple[list[float], dt] | None:
    """
    Busca a sequência inicial de dados do banco para o loop de forecast.
    
    Se 'city' for None, busca os dados agregados do estado.
    Se 'city' for preenchido, busca os dados da cidade específica.
    
    Adicionei 'order_by(desc())' para garantir que pegamos os dados mais recentes,
    o que torna a lógica de 'reverse()' segura.
    """
    query = (
        select(
            CasoCovid.datetime,
            func.sum(CasoCovid.new_confirmed).label("total_casos")
        )
        .where(CasoCovid.state == state_code)
        .where(CasoCovid.new_confirmed >= 0)
    )

    if city:
        query = query.where(CasoCovid.city == city)
    
    query = query.group_by(CasoCovid.datetime).order_by(CasoCovid.datetime.desc()).limit(seq_length)
    
    result = session.execute(query).all()

    if not result or len(result) < seq_length:
        return None
    
    result.reverse() 
    
    last_date = result[-1].datetime
    initial_sequence_values = [r.total_casos for r in result]
    
    return initial_sequence_values, last_date


def _generate_autoregressive_forecast(
    model, 
    scaler, 
    initial_sequence: list, 
    seq_length: int, 
    days: int, 
    last_date: dt
) -> list[dict]:
    """
    Função central que executa o loop de previsão autorregressivo.
    """
    current_sequence_scaled = scaler.transform(
        np.array(initial_sequence, dtype = np.float32).reshape(-1, 1)
    )

    forecast = []
    for i in range(days):
        with torch.no_grad():
            input_tensor = torch.from_numpy(current_sequence_scaled).float().view(1, seq_length, 1)
            pred_scaled = model(input_tensor).cpu().numpy()

        pred_real = scaler.inverse_transform(pred_scaled)[0][0]
        pred_real = max(0, pred_real) 

        forecast.append({
            "date": (last_date + timedelta(days = i + 1)).strftime("%Y-%m-%d"),
            "predicted_value": round(float(pred_real), 2)
        })

        current_sequence_scaled = np.vstack((
            current_sequence_scaled[1:],  
            pred_scaled.reshape(1, 1)     
        ))
        
    return forecast




def get_forecast_for_entire_state(state_code: str, days: int) -> dict:
    """
    Retorna a previsão multi-step para N dias do estado inteiro (agregado).
    (Agora usa as funções auxiliares)
    """
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Modelo não encontrado para {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    with sync_engine.connect() as session:
        sequence_data = _get_initial_sequence(session, state_code, seq_length, city = None)
        
        if not sequence_data:
            return {"error": f"Dados insuficientes para {state_code}."}
        
        initial_sequence, last_date = sequence_data
        
        forecast = _generate_autoregressive_forecast(
            model, scaler, initial_sequence, seq_length, days, last_date
        )

    return {
        "state": state_code,
        "model_run_id": run_id,
        "forecast": forecast
    }


def get_forecast_for_city(state_code: str, city: str, days: int) -> dict:
    """
    Gera previsões multi-step para uma cidade específica de um estado.
    (Agora usa as funções auxiliares)
    """
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Modelo não encontrado para {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    with sync_engine.connect() as session:
        sequence_data = _get_initial_sequence(session, state_code, seq_length, city = city)

        if not sequence_data:
            return {"error": f"Dados insuficientes para {city}, {state_code}."}
        
        initial_sequence, last_date = sequence_data

        forecast = _generate_autoregressive_forecast(
            model, scaler, initial_sequence, seq_length, days, last_date
        )

    return {
        "state": state_code,
        "city": city,
        "model_run_id": run_id,
        "forecast": forecast
    }


def get_forecast_for_state(state_code: str, days: int) -> dict:
    """
    Gera previsões multi-step para TODAS as cidades de um estado.
    (Esta era a função duplicada - agora é a única e usa os auxiliares)
    """
    artifacts = _load_model_from_mlflow(state_code)
    if not artifacts:
        return {"error": f"Modelo não encontrado para {state_code}"}

    model, scaler = artifacts["model"], artifacts["scaler"]
    seq_length, run_id = artifacts["seq_length"], artifacts["run_id"]

    forecasts_by_city = {}
    with sync_engine.connect() as session:
        cities_query = select(CasoCovid.city).where(CasoCovid.state == state_code).distinct()
        cities = [r.city for r in session.execute(cities_query).all()]

        if not cities:
            return {"error": f"Nenhuma cidade encontrada para {state_code}"}

        for city in cities:
            sequence_data = _get_initial_sequence(session, state_code, seq_length, city = city)
            
            if not sequence_data:
                print(f"WARNING: Dados insuficientes para {city} (necessário: {seq_length})")
                continue
            
            initial_sequence, last_date = sequence_data
            
            city_forecast = _generate_autoregressive_forecast(
                model, scaler, initial_sequence, seq_length, days, last_date
            )
            
            forecasts_by_city[str(city)] = city_forecast

    return {
        "state": state_code,
        "model_run_id": run_id,
        "forecasts": forecasts_by_city
    }

def get_forecast_with_confidence(state_code: str, days: int, confidence: float = 0.95) -> dict:
    """
    Gera a previsão para o estado agregado com intervalos de confiança.
    Esta função usa a 'get_forecast_for_entire_state' refatorada.
    """
    base_forecast = get_forecast_for_entire_state(state_code, days)
    if "forecast" not in base_forecast:
        return {"error": f"Falha ao gerar previsão para {state_code}"}

    predictions = np.array([item["predicted_value"] for item in base_forecast["forecast"]])
    
    std_dev = np.maximum(predictions * 0.05, 1.0)
    z_score = norm.ppf(1 - (1 - confidence)/2)

    forecast_with_ci = []
    for i, item in enumerate(base_forecast["forecast"]):
        forecast_with_ci.append({
            "date": item["date"],
            "predicted_mean": round(float(predictions[i]), 2),
            "lower_bound": round(max(0, predictions[i] - z_score * std_dev[i]), 2),
            "upper_bound": round(predictions[i] + z_score * std_dev[i], 2)
        })

    return {
        "state": state_code,
        "model_run_id": base_forecast["model_run_id"],
        "confidence_level": confidence,
        "forecast_with_confidence": forecast_with_ci
    }