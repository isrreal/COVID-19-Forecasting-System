FROM ghcr.io/mlflow/mlflow:v2.7.1

RUN pip install --no-cache-dir psycopg2-binary

RUN mkdir -p /mlflow/artifacts /mlflow/backend \
    && chmod 777 /mlflow/artifacts /mlflow/backend

WORKDIR /mlflow
