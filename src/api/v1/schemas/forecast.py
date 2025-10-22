from pydantic import BaseModel
from typing import List, Dict
from datetime import date

class PredictionRequest(BaseModel):
    sequence: List[float]

class ForecastItem(BaseModel):
    date: date
    predicted_value: float

class ForecastResponseByCity(BaseModel):
    state: str
    model_run_id: str
    forecasts: Dict[str, List[ForecastItem]]  

class ForecastResponse(BaseModel):
    state: str
    model_run_id: str
    forecast: List[ForecastItem]

class PredictionResponse(BaseModel):
    state: str
    model_run_id: str
    prediction: float