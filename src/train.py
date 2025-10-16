import os
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.data
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import select
from database import sync_engine
from src.models.neural_networks import CovidPredictorLSTM, CovidPredictorPLE 
from src.models.casos_covid import CasoCovid

def experiments_settings(state: str):
    """Define as configurações do experimento para AMBOS os modelos."""
    experiment_description = (
        f"Comparação de modelos (LSTM vs PLE) para forecasting de COVID-19 em {state}."
    )
    experiment_tags = {
        "project_name": "covid-19-forecasting-comparison",
        "state": state,
        "team": "leggen-assis-ml",
        "project_quarter": "Q4-2025",
        "mlflow.note.content": experiment_description,
    }

    lstm_search_space = {
        'model_type': ['LSTM'],
        'learning_rate': [0.001, 0.005],
        'hidden_size': [50, 60],
        'n_layers': [2, 5], 
        'sequence_length': [14, 30],
    }
    ple_search_space = {
        'model_type': ['PLE'],
        'learning_rate': [0.001, 0.005],
        'hidden_size': [50, 60],
        'num_experts': [3, 5],       
        'num_ple_layers': [2, 4],   
        'sequence_length': [14, 30],
    }

    base_params = {'epochs': [30], 'batch_size': [50]}
    
    lstm_param_grid = list(ParameterGrid({**lstm_search_space, **base_params}))
    ple_param_grid = list(ParameterGrid({**ple_search_space, **base_params}))
    
    param_grid = lstm_param_grid + ple_param_grid
    
    return {"tags": experiment_tags, "param_grid": param_grid}

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"train_rmse": rmse, "train_mae": mae}

def train_model(params: dict, state: str, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, scaler, dataset):
    """Função genérica para treinar um modelo (LSTM ou PLE)."""
    
    model_type = params['model_type']
    
    if model_type == 'LSTM':
        run_name = f"LSTM_LR_{params['learning_rate']}_HS_{params['hidden_size']}_NL_{params['n_layers']}"
    else:
        run_name = f"PLE_LR_{params['learning_rate']}_HS_{params['hidden_size']}_E_{params['num_experts']}"

    with mlflow.start_run(run_name = run_name) as run:
        run_id = run.info.run_id
        print(f"\n--- Iniciando nova Run ({model_type}): {run_name} ---")

        mlflow.log_input(dataset = dataset, context = "training")
        mlflow.set_tags({
            "model_type": model_type, 
            "Optimizer": "Adam", "run_purpose": "Grid Search Comparison",
            "target_variable": "new_confirmed", "feature_set": "v_univariate",
            "developer": "leggen-assis", "data_source_table": "casos_covid",
            "data_source_filter": f"state='{state}'"
        })
        mlflow.log_params(params)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size = params["batch_size"], shuffle = True
        )

        if model_type == 'LSTM':
            model = CovidPredictorLSTM(
                n_features = 1,
                hidden_size = params['hidden_size'],
                n_layers = params["n_layers"]
            )
        elif model_type == 'PLE':
            model = CovidPredictorPLE(
                n_features = 1,
                hidden_size = params['hidden_size'],
                num_experts = params['num_experts'],
                num_layers = params['num_ple_layers'] 
            )
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])

        print("Iniciando o treinamento do modelo...")
        y_true_inversed, y_pred_inversed, epoch_loss, metrics = None, None, None, None
        for epoch in range(params['epochs']):
            model.train()
            for seqs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                all_outputs_scaled = model(X_train_tensor)
                epoch_loss = criterion(all_outputs_scaled, y_train_tensor).item()
                y_true_inversed = scaler.inverse_transform(y_train_tensor.numpy())
                y_pred_inversed = scaler.inverse_transform(all_outputs_scaled.numpy())
                metrics = calculate_metrics(y_true_inversed, y_pred_inversed)
                
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.log_metrics({
                    "train_rmse": metrics["train_rmse"],
                    "train_mae": metrics["train_mae"]
                }, step = epoch)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{params["epochs"]}], Loss: {epoch_loss:.4f}, RMSE: {metrics["train_rmse"]:.2f}')
        
        print("Treinamento concluído. Salvando modelo e artefatos...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scaler_path = os.path.join(tmp_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.plot(y_true_inversed, label = 'Valores Reais')
            ax.plot(y_pred_inversed, label = 'Previsões', linestyle='--')
            ax.set_title(f'Previsões vs. Reais - {run_name}')
            ax.legend(); ax.grid(True)
            plot_path = os.path.join(tmp_dir, "predictions_vs_reals.png")
            fig.savefig(plot_path)
            plt.close(fig)

            mlflow.pytorch.log_model(
                pytorch_model = model,
                artifact_path = "model"
            )

            mlflow.log_artifact(scaler_path, artifact_path = "preprocessor")
            mlflow.log_artifact(plot_path, artifact_path = "plots")


        print(f"Modelo e artefatos salvos com sucesso para a run: {run_id}")


def run_experiments(state: str):
    """Orquestra todo o processo de experimentação para um único estado."""
    print(f"\n{'='*60}\nIniciando script de experimentação para o estado: {state}\n{'='*60}")
    
    settings = experiments_settings(state)
    
    print(f"Conectando ao banco de dados para buscar dados de {state}...")
    with sync_engine.connect() as conn:
        query = select(CasoCovid.datetime, CasoCovid.new_confirmed).where(CasoCovid.state == state)
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns = result.keys())
    
    if df.empty:
        print(f"Nenhum dado encontrado para o estado {state}. Pulando.")
        return
    
    df = df.sort_values('datetime').rename(columns = {'datetime': 'date'})
    print(f"Dados carregados com sucesso. {len(df)} registros encontrados.")
    
    mlflow_dataset = mlflow.data.from_pandas(df, name = f"casos_covid_{state.lower()}_train")
    
    experiment_name = f"Covid Forecasting Comparison - {state}"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(settings['tags'])
    
    time_series_raw = df['new_confirmed'].values.astype(float)
    scaler = MinMaxScaler(feature_range = (0, 1))
    time_series_scaled = scaler.fit_transform(time_series_raw.reshape(-1, 1))
    
    print(f"\nIniciando Grid Search com {len(settings['param_grid'])} combinações...")
    for idx, params in enumerate(settings['param_grid'], 1):
        print(f"\n[{idx}/{len(settings['param_grid'])}] Testando combinação: {params}")
        
        seq_length = params['sequence_length']
        X_train, y_train = create_sequences(time_series_scaled, seq_length)
        
        if len(X_train) == 0:
            print(f"Não foi possível criar sequências. Pulando.")
            continue
        
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        
        train_model(
            params, state, X_train_tensor, y_train_tensor, scaler, mlflow_dataset
        )
    
    print(f"\nTodos os experimentos para o estado {state} foram concluídos!")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    states_to_train = ["CE", "SP"] 
    
    for state_code in states_to_train:
        run_experiments(state = state_code)
    
    print(f"\n{'='*60}\n Processo finalizado com sucesso!\n{'='*60}")

