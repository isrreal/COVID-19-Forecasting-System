import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.v1.endpoints.forecast import router as forecast_router

app = FastAPI()
app.include_router(forecast_router, prefix="/forecast")

client = TestClient(app)

FORECAST_ITEM = {"date": "2025-10-18", "predicted_value": 100.0}
CONFIDENCE_ITEM = {
    "date": "2025-10-18",
    "predicted_mean": 100.0,
    "lower_bound": 90.0,
    "upper_bound": 110.0
}


# ==========================================================
# GET /forecast/state/{state_code}
# ==========================================================

def test_forecast_entire_state_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={"state": "CE", "model_run_id": "abc", "forecast": [FORECAST_ITEM]}
    )
    response = client.get("/forecast/state/CE")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "CE"
    assert len(data["forecast"]) == 1


def test_forecast_entire_state_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={}
    )
    response = client.get("/forecast/state/XX")
    assert response.status_code == 404
    assert "Forecast not available" in response.json()["detail"]


def test_forecast_entire_state_invalid_state_code():
    response = client.get("/forecast/state/CCC")
    assert response.status_code == 422


def test_forecast_entire_state_days_out_of_range():
    assert client.get("/forecast/state/CE?days=0").status_code == 422
    assert client.get("/forecast/state/CE?days=31").status_code == 422


def test_forecast_entire_state_default_days(mocker):
    mock = mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={"state": "CE", "model_run_id": "abc", "forecast": [FORECAST_ITEM]}
    )
    client.get("/forecast/state/CE")
    mock.assert_called_once_with("CE", 7)


# ==========================================================
# GET /forecast/state/{state_code}/confidence
# ==========================================================

def test_forecast_confidence_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_with_confidence",
        return_value={
            "state": "CE",
            "model_run_id": "abc",
            "confidence_level": 0.95,
            "forecast_with_confidence": [CONFIDENCE_ITEM]
        }
    )
    response = client.get("/forecast/state/CE/confidence")
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_level"] == 0.95
    assert len(data["forecast_with_confidence"]) == 1


def test_forecast_confidence_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_with_confidence",
        return_value={}
    )
    response = client.get("/forecast/state/XX/confidence")
    assert response.status_code == 404


def test_forecast_confidence_param_out_of_range():
    assert client.get("/forecast/state/CE/confidence?confidence=0.3").status_code == 422
    assert client.get("/forecast/state/CE/confidence?confidence=1.0").status_code == 422


# ==========================================================
# GET /forecast/cities/{state_code}
# ==========================================================

def test_forecast_all_cities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value={
            "state": "CE",
            "model_run_id": "abc",
            "forecasts": {"fortaleza": [FORECAST_ITEM]}
        }
    )
    response = client.get("/forecast/cities/CE")
    assert response.status_code == 200
    assert "fortaleza" in response.json()["forecasts"]


def test_forecast_all_cities_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value={"forecasts": {}}
    )
    response = client.get("/forecast/cities/XX")
    assert response.status_code == 404
    assert "No forecast found" in response.json()["detail"]


# ==========================================================
# GET /forecast/city/{state_code}/{city_name}
# ==========================================================

def test_forecast_specific_city_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_city",
        return_value={
            "state": "CE",
            "city": "fortaleza",
            "model_run_id": "abc",
            "forecast": [FORECAST_ITEM]
        }
    )
    response = client.get("/forecast/city/CE/Fortaleza")
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "fortaleza"
    assert len(data["forecast"]) == 1


def test_forecast_specific_city_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_city",
        return_value={}
    )
    response = client.get("/forecast/city/CE/Fortaleza")
    assert response.status_code == 404
    assert "No forecast found" in response.json()["detail"]


def test_forecast_specific_city_normalizes_accents(mocker):
    mock = mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_city",
        return_value={
            "state": "CE",
            "city": "sobral",
            "model_run_id": "abc",
            "forecast": [FORECAST_ITEM]
        }
    )
    client.get("/forecast/city/CE/Sóbral")
    assert mock.call_args[0][1] == "sobral"
