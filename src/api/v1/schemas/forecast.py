from pydantic import BaseModel
from datetime import date


class PredictionRequest(BaseModel):
    sequence: list[float]

class ForecastItem(BaseModel):
    date: date
    predicted_value: float

class ForecastResponseByCity(BaseModel):
    state: str
    model_run_id: str
    forecasts: dict[str, list[ForecastItem]]

class ForecastResponse(BaseModel):
    state: str
    model_run_id: str
    forecast: list[ForecastItem]

class PredictionResponse(BaseModel):
    state: str
    model_run_id: str
    prediction: float