from fastapi import FastAPI
from src.api.v1.api import api_router

app = FastAPI(
    title = "API de Forecasting de COVID-19",
    version = "1.0.0",
    description = "Uma API para servir previsões de novos casos de COVID-19 para estados do Brasil."
)

app.include_router(api_router, prefix = "/api/v1")

@app.get("/", tags = ["Root"])
def read_root():
    """Endpoint raiz para verificar se a API está online."""
    return {"message": "Bem-vindo à API de Forecasting de COVID-19!"}