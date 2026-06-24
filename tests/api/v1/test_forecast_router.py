from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from src.api.v1.endpoints.forecast import router as forecast_router

app = FastAPI()
app.include_router(forecast_router, prefix="/forecast")

client = TestClient(app)

FORECAST_ITEM = {"date": "2025-10-18", "predicted_value": 100.0}
CONFIDENCE_ITEM = {
    "date": "2025-10-18",
    "predicted_mean": 100.0,
    "lower_bound": 90.0,
    "upper_bound": 110.0,
}


# ==========================================================
# GET /forecast/state/{state_code}
# ==========================================================


def test_forecast_entire_state_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={
            "state": "CE",
            "model_run_id": "abc",
            "forecast": [FORECAST_ITEM],
        },
    )
    response = client.get("/forecast/state/CE")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["state"] == "CE"
    assert len(data["forecast"]) == 1


def test_forecast_entire_state_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={},
    )
    response = client.get("/forecast/state/XX")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Forecast not available" in response.json()["detail"]


def test_forecast_entire_state_invalid_state_code():
    response = client.get("/forecast/state/CCC")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_forecast_entire_state_days_out_of_range():
    assert (
        client.get("/forecast/state/CE?days=0").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    assert (
        client.get("/forecast/state/CE?days=31").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )


def test_forecast_entire_state_default_days(mocker):
    mock = mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_entire_state",
        return_value={
            "state": "CE",
            "model_run_id": "abc",
            "forecast": [FORECAST_ITEM],
        },
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
            "forecast_with_confidence": [CONFIDENCE_ITEM],
        },
    )
    response = client.get("/forecast/state/CE/confidence")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["confidence_level"] == 0.95
    assert len(data["forecast_with_confidence"]) == 1


def test_forecast_confidence_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_with_confidence",
        return_value={},
    )
    response = client.get("/forecast/state/XX/confidence")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_forecast_confidence_param_out_of_range():
    assert (
        client.get("/forecast/state/CE/confidence?confidence=0.3").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    assert (
        client.get("/forecast/state/CE/confidence?confidence=1.0").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# ==========================================================
# GET /forecast/municipalities/{state_code}
# ==========================================================


def test_forecast_all_municipalities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value={
            "state": "CE",
            "model_run_id": "abc",
            "forecasts": {"230440": [FORECAST_ITEM]},
        },
    )
    response = client.get("/forecast/municipalities/CE")
    assert response.status_code == status.HTTP_200_OK
    assert "230440" in response.json()["forecasts"]


def test_forecast_all_municipalities_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_state",
        return_value={"forecasts": {}},
    )
    response = client.get("/forecast/municipalities/XX")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "No forecast found" in response.json()["detail"]


# ==========================================================
# GET /forecast/municipality/{state_code}/{municipality_code}
# ==========================================================


def test_forecast_specific_municipality_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_municipality",
        return_value={
            "state": "CE",
            "municipality_code": 230440,
            "model_run_id": "abc",
            "forecast": [FORECAST_ITEM],
        },
    )
    response = client.get("/forecast/municipality/CE/230440")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["municipality_code"] == 230440
    assert len(data["forecast"]) == 1


def test_forecast_specific_municipality_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.forecast.get_forecast_for_municipality",
        return_value={},
    )
    response = client.get("/forecast/municipality/CE/230440")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "No forecast found" in response.json()["detail"]


def test_forecast_specific_municipality_invalid_code():
    response = client.get("/forecast/municipality/CE/not_a_number")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
