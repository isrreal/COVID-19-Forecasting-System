from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from database import get_sync_session
from src.api.v1.endpoints.stats import router as stats_router

app = FastAPI()
app.include_router(stats_router, prefix="/stats")

mock_session = MagicMock()
app.dependency_overrides[get_sync_session] = lambda: mock_session

client = TestClient(app)

SUMMARY = {
    "total_notifications": 1000,
    "total_deaths": 20,
    "hospitalization_rate": 0.15,
    "mortality_rate": 0.02,
}

MUNICIPALITY_STATS = {
    "municipality_code": 230440,
    "total_notifications": 500,
    "total_deaths": 10,
    "hospitalization_rate": 0.12,
    "mortality_rate": 0.02,
}

MUNICIPALITY_NOTIFICATION = {
    "municipality_code": 230440,
    "state_code": 23,
    "total_notifications": 500,
}

MUNICIPALITY_MORTALITY = {
    "municipality_code": 230440,
    "state_code": 23,
    "mortality_rate": 0.02,
    "total_deaths": 10,
    "total_notifications": 500,
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
    "metric": "daily_notifications",
    "mean": 100.0,
    "lower": 90.0,
    "upper": 110.0,
    "n": 1000,
}


# ==========================================================
# GET /stats/summary
# ==========================================================


def test_get_summary_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_summary_stats",
        return_value=SUMMARY,
    )
    response = client.get("/stats/summary")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_notifications"] == 1000
    assert data["total_deaths"] == 20


# ==========================================================
# GET /stats/municipality/{municipality_code}
# ==========================================================


def test_get_municipality_stats_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_municipality_stats",
        return_value=MUNICIPALITY_STATS,
    )
    response = client.get("/stats/municipality/230440")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["municipality_code"] == 230440


def test_get_municipality_stats_not_found(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_municipality_stats",
        return_value=None,
    )
    response = client.get("/stats/municipality/999999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_municipality_stats_invalid_code():
    response = client.get("/stats/municipality/not_a_number")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ==========================================================
# GET /stats/top-municipalities
# ==========================================================


def test_top_municipalities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_top_municipalities",
        return_value=[MUNICIPALITY_NOTIFICATION],
    )
    response = client.get("/stats/top-municipalities")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["data"]) == 1


def test_top_municipalities_service_error(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_top_municipalities",
        return_value={"error": "Could not retrieve data."},
    )
    response = client.get("/stats/top-municipalities")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ==========================================================
# GET /stats/chi-square/state-deaths
# ==========================================================


def test_chi_square_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.chi_square_state_deaths",
        return_value=CHI_SQUARE,
    )
    response = client.get("/stats/chi-square/state-deaths")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["reject_null_hypothesis"] is True
    assert data["test"] == "chi_square"


# ==========================================================
# GET /stats/most-deadly-municipalities
# ==========================================================


def test_most_deadly_municipalities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_most_deadly_municipalities",
        return_value=[MUNICIPALITY_MORTALITY],
    )
    response = client.get("/stats/most-deadly-municipalities")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["municipality_code"] == 230440


def test_most_deadly_municipalities_service_error(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_most_deadly_municipalities",
        return_value={"error": "Could not retrieve data."},
    )
    response = client.get("/stats/most-deadly-municipalities")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ==========================================================
# GET /stats/least-affected-municipalities
# ==========================================================


def test_least_affected_municipalities_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_least_affected_municipalities",
        return_value=[MUNICIPALITY_MORTALITY],
    )
    response = client.get("/stats/least-affected-municipalities")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["data"]) == 1


# ==========================================================
# GET /stats/confidence/daily-cases
# ==========================================================


def test_confidence_interval_daily_cases_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_confidence_interval_daily_cases",
        return_value=CONFIDENCE_INTERVAL,
    )
    response = client.get("/stats/confidence/daily-cases")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["lower"] < data["mean"] < data["upper"]


def test_confidence_interval_daily_cases_param_out_of_range():
    assert (
        client.get("/stats/confidence/daily-cases?confidence=0.7").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    assert (
        client.get("/stats/confidence/daily-cases?confidence=1.0").status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# ==========================================================
# GET /stats/confidence/daily-deaths
# ==========================================================


def test_confidence_interval_daily_deaths_success(mocker):
    mocker.patch(
        "src.api.v1.endpoints.stats.stats_service.get_confidence_interval_daily_deaths",
        return_value={**CONFIDENCE_INTERVAL, "metric": "daily_deaths"},
    )
    response = client.get("/stats/confidence/daily-deaths")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["metric"] == "daily_deaths"
