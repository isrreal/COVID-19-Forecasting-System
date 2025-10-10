FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir psycopg-binary

RUN mlflow server --host 127.0.0.1 --port 8080

