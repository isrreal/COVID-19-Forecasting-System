from fastapi import APIRouter, HTTPException, Query, Path
from unidecode import unidecode

from src.api.v1.schemas.forecast import (
    ForecastResponse, 
    ForecastResponseByCity
)
from src.api.v1.services.forecast_service import (
    get_forecast_for_state, 
    get_forecast_for_entire_state, 
    get_forecast_with_confidence, 
    get_forecast_for_city
)

router: APIRouter = APIRouter()

@router.get(
    "/state/{state_code}", 
    response_model = ForecastResponse,
    summary = "Previsão multi-step para o estado inteiro (agregado)"
)
def forecast_entire_state(
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE"),
    days: int = Query(default = 7, ge = 1, le = 30)
):
    forecast = get_forecast_for_entire_state(state_code.upper(), days)
    if not forecast or "forecast" not in forecast:
        raise HTTPException(status_code = 404, detail = f"Previsão não disponível para {state_code}")
    return forecast


@router.get(
    "/state/{state_code}/confidence", 
    summary = "Previsão multi-step com intervalo de confiança para o estado agregado"
)
def forecast_state_with_confidence(
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE"),
    days: int = Query(default = 7, ge = 1, le = 30),
    confidence: float = Query(default = 0.95, ge = 0.5, le = 0.99)
):
    forecast_ci = get_forecast_with_confidence(state_code.upper(), days, confidence)
    if not forecast_ci or "forecast_with_confidence" not in forecast_ci:
        raise HTTPException(status_code = 404, detail = f"Previsão não disponível para {state_code}")
    return forecast_ci


@router.get(
    "/cities/{state_code}", 
    response_model = ForecastResponseByCity,
    summary = "Previsão multi-step para todas as cidades de um estado"
)
def forecast_all_cities(
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE"),
    days: int = Query(default = 7, ge = 1, le = 30)
):
    forecast = get_forecast_for_state(state_code.upper(), days)
    if not forecast or not forecast.get("forecasts"):
        raise HTTPException(status_code = 404, detail = f"Nenhuma previsão encontrada para {state_code}")
    return forecast


@router.get(
    "/city/{state_code}/{city_name}", 
    summary = "Previsão multi-step para uma cidade específica de um estado"
)
def forecast_specific_city(
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE"),
    city_name: str = Path(min_length = 1, example = "Fortaleza"),
    days: int = Query(default = 7, ge = 1, le = 30)
):
    normalized_city_name = unidecode(city_name)
    forecast = get_forecast_for_city(state_code.upper(), normalized_city_name.lower(), days)
    if not forecast or "forecast" not in forecast:
        raise HTTPException(status_code = 404, detail = f"Nenhuma previsão encontrada para {city_name} ({state_code})")
    return forecast
