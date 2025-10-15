from fastapi import APIRouter, HTTPException, Query, Path

from src.api.v1.schemas.forecast import (
    PredictionRequest, 
    ForecastResponse, 
    PredictionResponse
)

from src.api.v1.services.forecast_service import get_prediction_for_state, get_forecast_for_state

router: APIRouter = APIRouter()

@router.post(
    "/predict/{state_code}", 
    response_model = PredictionResponse, 
    summary = "Obtém uma previsão para o próximo dia"
)
def predict_next_day(
    payload: PredictionRequest,
    state_code: str = Path(
                        min_length = 2,
                        max_length = 2,
                        example = "CE",
                        description = "Sigla do estado (UF)"
                    )
):
    """
    Recebe uma sequência de dados de novos casos confirmados e retorna a previsão
    para o dia seguinte.
    """
    if len(payload.sequence) < 14: 
        raise HTTPException(status_code = 400, detail = "A sequência deve conter pelo menos 14 dias de dados.")
    
    prediction = get_prediction_for_state(state_code.upper(), payload.sequence)
    return prediction

@router.get(
    "/{state_code}", 
    response_model = ForecastResponse,
    summary = "Gera uma previsão para os próximos N dias"
)
def get_forecast(
    state_code: str = Path(
        min_length = 2,
        max_length = 2, 
        example = "SP", 
        description = "Sigla do estado (UF)"
    ),
    days: int = Query(
        default = 7, 
        ge = 1, 
        le = 30, 
        description = "Número de dias para prever no futuro."
    )
):
    """
    Busca os dados mais recentes do banco de dados e gera uma previsão para os próximos `days`.
    """
    forecast_data = get_forecast_for_state(state_code.upper(), days)

    if not forecast_data:
        raise HTTPException(status_code = 404, detail = f"Modelo para o estado {state_code} não encontrado.")
    
    return forecast_data