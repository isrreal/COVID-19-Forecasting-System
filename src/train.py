import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

from sqlalchemy import select
from database import sync_engine
from src.models.neural_networks import CovidPredictorLSTM
from src.models.casos_covid import CasoCovid  

def experiments_settings():
    experiment_description = (
        "Projeto de forecasting para a série temporal de novos casos de COVID-19 no estado do Ceará (CE). "
        "Este experimento testa diferentes hiperparâmetros para um modelo LSTM."
    )
    experiment_tags = {
        "project_name": "covid-19-forecasting",
        "state": "CE",
        "team": "leggen-assis-ml",
        "project_quarter": "Q4-2025",
        "mlflow.note.content": experiment_description,
    }
    search_space = {
        'learning_rate': [0.001, 0.0005],
        'hidden_size': [50, 64],
        'epochs': [30, 50],
        'sequence_length': [14]
    }
    param_grid = list(ParameterGrid(search_space))
    return {"tags": experiment_tags, "param_grid": param_grid}

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def calculate_metrics(y_true_scaled, y_pred_scaled, scaler):
    y_true_inversed = scaler.inverse_transform(y_true_scaled)
    y_pred_inversed = scaler.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_true_inversed, y_pred_inversed))
    mae = mean_absolute_error(y_true_inversed, y_pred_inversed)
    return {"train_rmse": rmse, "train_mae": mae}


def train_single_model(params: dict, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, scaler, dataset: PandasDataset):
    run_name = f"LR_{params['learning_rate']}_HS_{params['hidden_size']}_EP_{params['epochs']}"
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Iniciando nova Run: {run_name} ---")
        mlflow.log_input(dataset=dataset, context="training")
        mlflow.set_tags({
            "model_type": "LSTM",
            "run_purpose": "Grid Search",
            "target_variable": "new_confirmed",
            "feature_set": "v1_univariate",
            "developer": "leggen-assis",
            "data_source_table": "casos_covid",
            "data_source_filter": "state='CE'"
        })
        mlflow.log_params(params)

        BATCH_SIZE = 16
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=BATCH_SIZE, shuffle=True)

        model = CovidPredictorLSTM(n_features=1, hidden_size=params['hidden_size'], n_layers=2)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        for epoch in range(params['epochs']):
            model.train()
            for seqs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(seqs), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                all_outputs = model(X_train_tensor)
                epoch_loss = criterion(all_outputs, y_train_tensor).item()
                metrics = calculate_metrics(y_train_tensor.numpy(), all_outputs.numpy(), scaler)

            mlflow.log_metric("train_loss_epoch", epoch_loss, step=epoch)
            mlflow.log_metrics(metrics, step=epoch)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{params["epochs"]}], Loss: {epoch_loss:.4f}, RMSE: {metrics["train_rmse"]:.2f}, MAE: {metrics["train_mae"]:.2f}')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(scaler.inverse_transform(y_train_tensor.numpy()), label='Valores Reais')
        ax.plot(scaler.inverse_transform(all_outputs.numpy()), label='Previsões', linestyle='--')
        ax.set_title(f'Previsões vs. Reais - Run: {run_name}')
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

def run_experiments():
    print("Iniciando o script de experimentação...")
    settings = experiments_settings()

    print("Conectando ao banco de dados via SQLAlchemy ORM...")
    with sync_engine.connect() as conn:
        query = select(CasoCovid.datetime, CasoCovid.new_confirmed).where(CasoCovid.state == 'CE')
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    df = df.sort_values('datetime').rename(columns={'datetime': 'date'})
    print(f"Dados carregados com sucesso. {len(df)} registros encontrados.")

    mlflow_dataset = mlflow.data.from_pandas(df, name="casos_covid_ce_train")
    time_series = df['new_confirmed'].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1))

    mlflow.set_experiment("Covid Forecasting")
    mlflow.set_experiment_tags(settings['tags'])

    for params in settings['param_grid']:
        X_train, y_train = create_sequences(time_series_scaled, params['sequence_length'])
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        train_single_model(params, X_train_tensor, y_train_tensor, scaler, mlflow_dataset)

    print("\nTodos os experimentos concluídos! Acesse a UI do MLflow em http://localhost:5001.")

if __name__ == "__main__":
    run_experiments()
