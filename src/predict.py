import os
import mlflow
import joblib
import torch
import pandas as pd
from mlflow.tracking import MlflowClient 

from database import sync_engine

BEST_RUN_ID = "2425a5f2d16c4974b9026c6de4f61796" 
SEQ_LENGTH = 14 

def predict_next_day(run_id: str):
    """
    Carrega o melhor modelo do MLflow e faz uma previsão para o próximo dia.
    """
    if run_id == "COLE_O_SEU_MELHOR_RUN_ID_AQUI":
        raise ValueError("ERRO: Por favor, atualize a variável BEST_RUN_ID com um Run ID válido.")
        
    print(f"Iniciando previsão usando o modelo da Run ID: {run_id}")

    print("Carregando o scaler do MLflow...")
    try:
        client = MlflowClient()

        local_scaler_dir = client.download_artifacts(run_id, "scaler", "/tmp")
        scaler_path = os.path.join(local_scaler_dir, f"scaler_{run_id}.gz")

        print(f"Scaler baixado em: {scaler_path}")
        scaler = joblib.load(scaler_path)

    except Exception as e:
        print(f"Erro ao carregar o scaler: {e}")
        return None 

    print("Carregando o modelo PyTorch do MLflow...")
    try:
        model_uri = f"runs:/{run_id}/pytorch-model"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval() 
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

    print(f"Buscando os últimos {SEQ_LENGTH} dias de dados do banco...")
    query = f"SELECT new_confirmed FROM casos_covid WHERE state='CE' ORDER BY datetime DESC LIMIT {SEQ_LENGTH}"
    latest_data_df = pd.read_sql(query, sync_engine)
    
    if len(latest_data_df) < SEQ_LENGTH:
        print(f"Erro: Foram encontrados apenas {len(latest_data_df)} registros. São necessários {SEQ_LENGTH}.")
        return None

    last_n_days_data = latest_data_df['new_confirmed'].values[::-1]
    
    data_scaled = scaler.transform(last_n_days_data.reshape(-1, 1))
    
    input_tensor = torch.from_numpy(data_scaled).float().view(1, SEQ_LENGTH, 1)

    print("Realizando a predição...")
    with torch.no_grad(): 
        prediction_scaled = model(input_tensor)

    prediction_inversed = scaler.inverse_transform(prediction_scaled.numpy())
    
    predicted_cases = prediction_inversed[0][0]
    print("-" * 50)
    print(f"Previsão de novos casos para o próximo dia: {predicted_cases:.2f}")
    print("-" * 50)
    
    return predicted_cases


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    
    predict_next_day(run_id = BEST_RUN_ID)