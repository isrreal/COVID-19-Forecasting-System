from fastapi import APIRouter

from api.v1.endpoints import forecast

api_router: APIRouter = APIRouter()

api_router.include_router(forecast.router, prefix = "/forecast", tags = ["Forecasting"])
