from fastapi import FastAPI
from src.api.v1.api import api_router

app = FastAPI(
    title="Dengue Forecasting API",
    version="2.0.0",
    description="API for serving dengue case forecasts across Brazilian states using SINAN data.",
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Dengue Forecasting API!"}
