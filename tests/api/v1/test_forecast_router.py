from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest
from src.api.v1.endpoints.forecast import router as forecast_router

app = FastAPI()
app.include_router(forecast_router, prefix = "/api/v1/forecast")

client = TestClient(app)


def test_predict_next_day_success(mocker):
    """
    Testa o cenário de sucesso para a rota de predição.
    """
    mock_response = {
        "state": "CE",
        "model_run_id": "test_run_id_123",
        "prediction": 950.5
    }
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_prediction_for_state",
        return_value = mock_response
    )

    request_payload = {
        "sequence": [100] * 30
    }

    response = client.post("/api/v1/forecast/predict/CE", json=request_payload)

    assert response.status_code == 200
    assert response.json() == mock_response


def test_predict_next_day_model_not_found(mocker):
    """
    Testa o cenário onde o modelo para o estado não é encontrado (404).
    """
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_prediction_for_state",
        return_value = None
    )

    request_payload = {"sequence": [1] * 7}

    response = client.post("/api/v1/forecast/predict/XX", json = request_payload)

    assert response.status_code == 404
    assert "Modelo para o estado XX não encontrado" in response.json()["detail"]


def test_predict_next_day_invalid_sequence(mocker):
    """
    Testa o cenário onde a lógica de serviço levanta um ValueError (400).
    """
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_prediction_for_state",
        side_effect = ValueError("O tamanho da sequência é inválido.")
    )

    request_payload = {"sequence": [1, 2, 3]}

    response = client.post("/api/v1/forecast/predict/SP", json = request_payload)

    assert response.status_code == 400
    assert "O tamanho da sequência é inválido" in response.json()["detail"]


def test_predict_invalid_state_code_format():
    """
    Testa a validação do FastAPI para um state_code inválido (422).
    """
    request_payload = {"sequence": [1] * 7}

    response = client.post("/api/v1/forecast/predict/Ceará", json = request_payload)

    assert response.status_code == 422


def test_get_forecast_success(mocker):
    """
    Testa o cenário de sucesso para a rota de forecast.
    """
    mock_response = {
        "state": "SP",
        "model_run_id": "test_run_id_456",
        "forecast": [
            {"date": "2025-10-18", "predicted_value": 1200.0},
            {"date": "2025-10-19", "predicted_value": 1250.5}
        ]
    }
    mock_get_forecast = mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value = mock_response
    )

    response = client.get("/api/v1/forecast/SP?days=2")

    assert response.status_code == 200
    assert response.json() == mock_response
    mock_get_forecast.assert_called_once_with("SP", 2)


def test_get_forecast_default_days(mocker):
    """
    Testa se o valor padrão de `days` (7) é usado corretamente.
    """
    mock_get_forecast = mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value = {"state": "RJ", "model_run_id": "xyz", "forecast": []}
    )

    response = client.get("/api/v1/forecast/RJ")

    assert response.status_code == 200
    mock_get_forecast.assert_called_once_with("RJ", 7)


def test_get_forecast_model_not_found(mocker):
    """
    Testa o cenário onde o modelo para o estado não é encontrado (404).
    """
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value = None
    )

    response = client.get("/api/v1/forecast/AM")

    assert response.status_code == 404
    assert "Modelo para o estado AM não encontrado" in response.json()["detail"]


def test_get_forecast_invalid_days_parameter():
    """
    Testa a validação do FastAPI para o parâmetro `days` fora do range (422).
    """
    response_less = client.get("/api/v1/forecast/MG?days=0")
    assert response_less.status_code == 422

    response_greater = client.get("/api/v1/forecast/MG?days=31")
    assert response_greater.status_code == 422
