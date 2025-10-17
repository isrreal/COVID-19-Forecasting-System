from fastapi import APIRouter
from src.api.v1.endpoints import forecast, stats

api_router: APIRouter = APIRouter()

api_router.include_router(
    forecast.router,
    prefix = "/forecast",
    tags = ["Forecasting"]
)

api_router.include_router(
    stats.router,
    prefix = "/stats",
    tags = ["Statistics"]
)
