FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.22.5 psycopg2-binary

RUN mkdir -p /mlflow/artifacts /mlflow/backend \
    && chmod 777 /mlflow/artifacts /mlflow/backend

WORKDIR /mlflow
