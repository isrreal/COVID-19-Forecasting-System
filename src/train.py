import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.data 
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import joblib
import matplotlib.pyplot as plt

from sqlalchemy import select
from database import sync_engine
from src.models.neural_networks import CovidPredictorLSTM
from src.models.casos_covid import CasoCovid

def experiments_settings_lstm(state: str):
    """Define as configurações do experimento para um estado específico."""

    experiment_description = (
        f"Projeto de forecasting para a série temporal de novos casos de COVID-19 no estado de {state}. "
        "Este experimento testa diferentes hiperparâmetros para um modelo LSTM."
    )
    experiment_tags = {
        "project_name": "covid-19-forecasting",
        "state": state,
        "team": "leggen-assis-ml",
        "project_quarter": "Q4-2025",
        "mlflow.note.content": experiment_description,
    }
    search_space = {
        'learning_rate': [0.001, 0.0005],
        'hidden_size': [50, 64, 100],
        'epochs': [30, 50, 100],
        'sequence_length': [14, 360],
        'batch_size': [10, 50, 100],
        'n_layers': [2, 10, 20, 30]
    }

    param_grid = list(ParameterGrid(search_space))
    return {"tags": experiment_tags, "param_grid": param_grid}

def create_sequences(data, seq_length):
    """Cria sequências de dados."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de regressão (RMSE e MAE). Agora aceita dados não escalonados."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"train_rmse": rmse, "train_mae": mae}


def train_lstm_model(params: dict,
        state: str,
        X_train_tensor: torch.Tensor,
        y_train_tensor: torch.Tensor,
        scaler,
        dataset
    ):
    """Função para treinar um único modelo LSTM."""

    run_name = f"LSTM_LR_{params['learning_rate']}_HS_{params['hidden_size']}_EP_{params['epochs']}"

    with mlflow.start_run(run_name = run_name):
        print(f"\n--- Iniciando nova Run (LSTM): {run_name} ---")
        mlflow.log_input(dataset = dataset, context = "training")
        
        mlflow.set_tags({
            "model_type": "LSTM",
            "Optimizer": "Adam",
            "run_purpose": "Grid Search",
            "target_variable": "new_confirmed",
            "feature_set": "v1_univariate",
            "developer": "leggen-assis",
            "data_source_table": "casos_covid",
            "data_source_filter": f"state='{state}'"
        })
        
        mlflow.log_params(params)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size = params["batch_size"],
            shuffle = True
        )

        model = CovidPredictorLSTM(
            n_features = 1,
            hidden_size = params['hidden_size'],
            n_layers = params["n_layers"]
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])

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

            mlflow.log_metric("train_loss_epoch", epoch_loss, step = epoch)
            mlflow.log_metrics(metrics, step = epoch)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{params["epochs"]}], Loss: {epoch_loss:.4f}, RMSE: {metrics["train_rmse"]:.2f}, MAE: {metrics["train_mae"]:.2f}')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_true_inversed, label = 'Valores Reais')
        ax.plot(y_pred_inversed, label = 'Previsões', linestyle='--')
        ax.set_title(f'Previsões vs. Reais (LSTM) - Run: {run_name}')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Novos Casos Confirmados')
        ax.legend()
        ax.grid(True)
        plot_path = f"/tmp/predictions_vs_reals_{mlflow.active_run().info.run_id}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path, "plots")
        os.remove(plot_path)
        scaler_path = f"/tmp/scaler_{mlflow.active_run().info.run_id}.gz"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "scaler")
        os.remove(scaler_path)
        mlflow.pytorch.log_model(model, "pytorch-model")

def run_experiments(state: str):
    """Orquestra todo o processo de experimentação para um único estado."""
    print(f"\n{'='*60}\nIniciando script de experimentação para o estado: {state}\n{'='*60}")
    
    settings = experiments_settings_lstm(state)
    print(f"Conectando ao banco de dados para buscar dados de {state}...")
    with sync_engine.connect() as conn:
        query = select(CasoCovid.datetime, CasoCovid.new_confirmed).where(CasoCovid.state == state)
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns = result.keys())
    
    if df.empty:
        print(f"Nenhum dado encontrado para o estado {state}. Pulando.")
        return
        
    df = df.sort_values('datetime').rename(columns={'datetime': 'date'})
    print(f"Dados carregados com sucesso. {len(df)} registros encontrados para {state}.")

    mlflow_dataset = mlflow.data.from_pandas(df, name = f"casos_covid_{state.lower()}_train")

    experiment_name = f"Covid Forecasting - {state}"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(settings['tags'])

    time_series_raw = df['new_confirmed'].values.astype(float)

    scaler = MinMaxScaler(feature_range = (0, 1))
    time_series_scaled = scaler.fit_transform(time_series_raw.reshape(-1, 1))

    for params in settings['param_grid']:
        seq_length_lstm = params['sequence_length']
        X_train_lstm, y_train_lstm = create_sequences(time_series_scaled, seq_length_lstm)
        X_train_tensor = torch.from_numpy(X_train_lstm).float()
        y_train_tensor = torch.from_numpy(y_train_lstm).float()

        train_lstm_model(
            params,
            state, 
            X_train_tensor, 
            y_train_tensor, 
            scaler, 
            mlflow_dataset
        )

    print(f"\nTodos os experimentos para o estado {state} foram concluídos!")

if __name__ == "__main__":
    states_to_train = ["CE"]#, "SP", "RJ", "PE"]
    for state_code in states_to_train:
        run_experiments(state = state_code)
    print(f"\n{'='*60}\nProcesso finalizado!\n{'='*60}")