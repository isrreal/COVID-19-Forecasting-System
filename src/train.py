import os
import pandas as pd
import mlflow
import torch

from database import sync_engine
from src.models.neural_networks import CovidPredictorLSTM

def train_model():
    """
    Função principal para treinar o modelo.
    """
    print("Iniciando o script de treinamento...")
    
    print("Conectando ao banco de dados para buscar os dados de treinamento...")
    df = pd.read_sql("SELECT * FROM casos_covid WHERE state='CE'", sync_engine)
    print(f"Dados carregados com sucesso. Encontrados {len(df)} registos para o estado CE.")
    
  
    X_train_tensor = torch.randn(100, 10, 1) 
    y_train_tensor = torch.randn(100, 1)


    mlflow.set_tracking_uri("http://127.0.0.1:8000")

    with mlflow.start_run():
        print("Iniciando run do MLflow...")
        
        # --- Hiperparâmetros ---
        N_FEATURES = 1 
        HIDDEN_SIZE = 50
        N_LAYERS = 2
        LEARNING_RATE = 0.001
        EPOCHS = 10

        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", N_LAYERS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)

        model = CovidPredictorLSTM(n_features=N_FEATURES, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("Iniciando o loop de treinamento...")
        for epoch in range(EPOCHS):
            outputs = model(X_train_tensor)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

        print("Treinamento concluído.")
        
        # --- Logging do Modelo ---
        mlflow.pytorch.log_model(model, "pytorch-model")
        print("Modelo logado no MLflow com sucesso!")
        print(f"Para ver os resultados, execute 'mlflow ui' na pasta do projeto.")

if __name__ == "__main__":
    train_model()