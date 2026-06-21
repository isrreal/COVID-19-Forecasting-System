import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from database import get_sync_session
from src.api.v1.endpoints.stats import router as stats_router

app = FastAPI()
app.include_router(stats_router, prefix="/stats")

mock_session = MagicMock()
app.dependency_overrides[get_sync_session] = lambda: mock_session

client = TestClient(app)

SUMMARY = {
    "total_records": 1000,
    "total_confirmed": 50000.0,
    "total_deaths": 1000.0,
    "avg_new_confirmed_per_day": 50.0,
    "avg_new_deaths_per_day": 1.0,
}

CITY_STATS = {
    "city": "fortaleza",
    "total_confirmed": 1000.0,
    "total_deaths": 20.0,
    "avg_new_confirmed": 10.0,
    "avg_new_deaths": 0.2,
}

CITY_CONFIRMED = {"city": "Fortaleza", "total_confirmed": 1000.0}

CITY_MORTALITY = {
    "city": "Fortaleza",
    "state": "CE",
    "mortality_rate": 0.02,
    "total_deaths": 20.0,
    "total_confirmed": 1000.0,
}

CHI_SQUARE = {
    "test": "chi_square",
    "null_hypothesis": "Death occurrence is independent of the state",
    "chi2_statistic": 10.5,
    "p_value": 0.001,
    "degrees_of_freedom": 26,
    "significance_level": 0.05,
    "reject_null_hypothesis": True,
    "interpretation": "There is a significant association.",
    "contingency_table": {"0": {"CE": 100}, "1": {"CE": 50}},
    "expected_frequencies": [[75.0, 75.0]],
}

CONFIDENCE_INTERVAL = {
    "metric": "new_confirmed",
    "mean": 100.0,
    "lower": 90.0,
    "upper": 110.0,
    "n": 1000,
}


# ==========================================================
# GET /stats/summary
# ==========================================================

def test_get_summary_success(mocker):
    mocker.patch("src.api.v1.endpoints.stats.stats_service.get_summary_stats", return_value=SUMMARY)
    response = client.get("/stats/summary")
    assert response.status_code == 200
    assert response.json()["total_records"] == 1000


# ==========================================================
# GET /stats/city/{city_name}/{state}
# ==========================================================

def test_get_city_stats_success(mocker):
    mocker.patch("src.api.v1.endpoints.stats.stats_service.get_city_stats", return_value=CITY_STATS)
    response = client.get("/stats/city/Fortaleza/CE")
    assert response.status_code == 200
    assert response.json()["city"] == "fortaleza"


def test_get_city_stats_invalid_city_name():
    response = client.get("/stats/city/F/CE")
    assert response.status_code == 422


# ==========================================================
# GET /stats/top-cities
# ==========================================================

def test_top_cities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_top_cities",
        return_value=[CITY_CONFIRMED]
    )
    response = client.get("/stats/top-cities")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_top_cities_service_error(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_top_cities",
        return_value={"error": "Could not retrieve data."}
    )
    response = client.get("/stats/top-cities")
    assert response.status_code == 500


# ==========================================================
# GET /stats/chi-square/state-deaths
# ==========================================================

def test_chi_square_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.chi_square_state_deaths",
        return_value=CHI_SQUARE
    )
    response = client.get("/stats/chi-square/state-deaths")
    assert response.status_code == 200
    data = response.json()
    assert data["reject_null_hypothesis"] is True
    assert data["test"] == "chi_square"


# ==========================================================
# GET /stats/most-deadly-cities
# ==========================================================

def test_most_deadly_cities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_most_deadly_cities",
        return_value=[CITY_MORTALITY]
    )
    response = client.get("/stats/most-deadly-cities")
    assert response.status_code == 200
    assert response.json()["data"][0]["city"] == "Fortaleza"


def test_most_deadly_cities_service_error(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_most_deadly_cities",
        return_value={"error": "Could not retrieve data."}
    )
    response = client.get("/stats/most-deadly-cities")
    assert response.status_code == 500


# ==========================================================
# GET /stats/least-affected-cities
# ==========================================================

def test_least_affected_cities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_least_affected_cities",
        return_value=[CITY_MORTALITY]
    )
    response = client.get("/stats/least-affected-cities")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


# ==========================================================
# GET /stats/confidence/cases
# ==========================================================

def test_confidence_interval_cases_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_confidence_interval_cases",
        return_value=CONFIDENCE_INTERVAL
    )
    response = client.get("/stats/confidence/cases")
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "new_confirmed"
    assert data["lower"] < data["mean"] < data["upper"]


def test_confidence_interval_cases_param_out_of_range():
    assert client.get("/stats/confidence/cases?confidence=0.7").status_code == 422
    assert client.get("/stats/confidence/cases?confidence=1.0").status_code == 422


# ==========================================================
# GET /stats/confidence/deaths
# ==========================================================

def test_confidence_interval_deaths_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_confidence_interval_deaths",
        return_value={**CONFIDENCE_INTERVAL, "metric": "new_deaths"}
    )
    response = client.get("/stats/confidence/deaths")
    assert response.status_code == 200
    assert response.json()["metric"] == "new_deaths"
