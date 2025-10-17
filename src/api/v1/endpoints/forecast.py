from fastapi import APIRouter, HTTPException, Query, Path

from src.api.v1.schemas.forecast import (
    PredictionRequest, 
    ForecastResponse, 
    PredictionResponse
)
from src.api.v1.services.forecast_service import get_prediction_for_state, get_forecast_for_state

router: APIRouter = APIRouter()

@router.post("/predict/{state_code}", response_model = PredictionResponse, summary = "Obtém uma previsão para o próximo dia")
def predict_next_day(
    payload: PredictionRequest,
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE")
):
    try:
        prediction = get_prediction_for_state(state_code.upper(), payload.sequence)
        if not prediction:
            raise HTTPException(status_code = 404, detail = f"Modelo para o estado {state_code} não encontrado.")
        return prediction

    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    except HTTPException:
        raise

    except Exception as e:
        print(f"Erro inesperado em predict_next_day: {e}")
        raise HTTPException(status_code = 500, detail="Ocorreu um erro interno ao processar a previsão.")



@router.get(
    "/{state_code}", 
    response_model = ForecastResponse,
    summary = "Gera uma previsão para os próximos N dias"
)
def get_forecast(
    state_code: str = Path(min_length = 2, max_length = 2, example = "CE"),
    days: int = Query(default = 7, ge = 1, le = 30)
):
    try:
        forecast_data = get_forecast_for_state(state_code.upper(), days)
        if not forecast_data:
            raise HTTPException(status_code = 404, detail = f"Modelo para o estado {state_code} não encontrado.")
        return forecast_data

    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))

    except HTTPException:
        raise

    except Exception as e:
        print(f"Erro inesperado em get_forecast: {e}")
        raise HTTPException(status_code = 500, detail = "Ocorreu um erro interno ao gerar a previsão.")

