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

from database import sync_engine
from src.models.neural_networks import CovidPredictorLSTM


def experiments_settings():
    """Define todas as configurações do experimento em um único lugar."""
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
    """Cria sequências de dados para a LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def calculate_metrics(y_true_scaled, y_pred_scaled, scaler):
    """Calcula métricas de regressão (RMSE e MAE) des-escalonando os dados."""
    y_true_inversed = scaler.inverse_transform(y_true_scaled)
    y_pred_inversed = scaler.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_true_inversed, y_pred_inversed))
    mae = mean_absolute_error(y_true_inversed, y_pred_inversed)
    return {"train_rmse": rmse, "train_mae": mae}


def train_single_model(params: dict, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, scaler, dataset: PandasDataset):
    """Função para treinar um único modelo com um conjunto de hiperparâmetros."""
    run_name = f"LR_{params['learning_rate']}_HS_{params['hidden_size']}_EP_{params['epochs']}"
    with mlflow.start_run(run_name = run_name):
        print(f"\n--- Iniciando nova Run: {run_name} ---")
        mlflow.log_input(dataset = dataset, context="training")
        run_tags = {
            "model_type": "LSTM", "run_purpose": "Grid Search", "target_variable": "new_confirmed",
            "feature_set": "v1_univariate", "developer": "leggen-assis", "data_source_table": "casos_covid",
            "data_source_filter": "state='CE'"
        }
        mlflow.set_tags(run_tags)
        mlflow.log_params(params)

        BATCH_SIZE = 16
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        model = CovidPredictorLSTM(n_features = 1, hidden_size = params['hidden_size'], n_layers = 2)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])

        print("Iniciando o loop de treinamento...")
        for epoch in range(params['epochs']):
            model.train()
            for seqs, labels in train_loader:
                outputs = model(seqs)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                all_outputs = model(X_train_tensor)
                epoch_loss = criterion(all_outputs, y_train_tensor).item()
                y_true_np = y_train_tensor.numpy()
                y_pred_np = all_outputs.numpy()
                metrics = calculate_metrics(y_true_np, y_pred_np, scaler)
            
            mlflow.log_metric("train_loss_epoch", epoch_loss, step = epoch)
            mlflow.log_metrics(metrics, step = epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{params["epochs"]}], Loss: {epoch_loss:.4f}, RMSE: {metrics["train_rmse"]:.2f}, MAE: {metrics["train_mae"]:.2f}')
        
        print("Treinamento concluído.")

        print("Gerando gráfico de previsões vs. valores reais...")
        model.eval()
        with torch.no_grad():
            final_preds_scaled = all_outputs.numpy()
            y_true_scaled = y_train_tensor.numpy()

            final_preds_inversed = scaler.inverse_transform(final_preds_scaled)
            y_true_inversed = scaler.inverse_transform(y_true_scaled)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_true_inversed, label = 'Valores Reais')
            ax.plot(final_preds_inversed, label = 'Previsões do Modelo', linestyle = '--')
            ax.set_title(f'Previsões vs. Reais - Run: {run_name}')
            ax.set_xlabel('Tempo (Amostras de Treino)')
            ax.set_ylabel('Novos Casos Confirmados')
            ax.legend()
            ax.grid(True)

            plot_path = f"/tmp/predictions_vs_reals_{mlflow.active_run().info.run_id}.png"
            fig.savefig(plot_path)
            plt.close(fig) 

            mlflow.log_artifact(plot_path, "plots")
            os.remove(plot_path)
            print("Gráfico logado como artefato com sucesso.")
        
        scaler_path = f"/tmp/scaler_{mlflow.active_run().info.run_id}.gz"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "scaler")
        os.remove(scaler_path)
        mlflow.pytorch.log_model(model, "pytorch-model")
        print("Modelo e scaler logados no MLflow com sucesso!")

def run_experiments():
    print("Iniciando o script de experimentação...")
    settings = experiments_settings()
    print("Conectando ao banco de dados...")
    query = "SELECT datetime, new_confirmed FROM casos_covid WHERE state='CE'"
    df = pd.read_sql(query, sync_engine, parse_dates = ['datetime'])
    df = df.sort_values(by = 'datetime')
    df.rename(columns = {'datetime': 'date'}, inplace = True)
    print(f"Dados carregados com sucesso. Encontrados {len(df)} registros.")
    
    source_description = f"tabela: casos_covid, filtro: state=CE"
    mlflow_dataset = mlflow.data.from_pandas(df, name = "casos_covid_ce_train")

    time_series = df['new_confirmed'].values.astype(float)
    scaler = MinMaxScaler(feature_range = (0, 1))
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1))
    
    experiment_name = "Covid Forecasting"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(settings['tags'])
    
    for params in settings['param_grid']:
        seq_length = params['sequence_length']
        X_train, y_train = create_sequences(time_series_scaled, seq_length)
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        train_single_model(params, X_train_tensor, y_train_tensor, scaler, mlflow_dataset)

    print("\nTodos os experimentos foram concluídos!")
    print(f"Acesse a UI do MLflow em http://localhost:5001 para comparar os resultados.")

if __name__ == "__main__":
    run_experiments()