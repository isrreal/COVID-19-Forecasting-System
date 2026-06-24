import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _get(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_BASE_URL}/api/v1{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_summary() -> dict:
    return _get("/stats/summary")


def get_top_municipalities(limit: int = 10) -> list:
    return _get("/stats/top-municipalities", {"limit": limit})["data"]


def get_most_deadly_municipalities(limit: int = 10) -> list:
    return _get("/stats/most-deadly-municipalities", {"limit": limit})["data"]


def get_least_affected_municipalities(limit: int = 10) -> list:
    return _get("/stats/least-affected-municipalities", {"limit": limit})["data"]


def get_confidence_interval_daily_cases(confidence: float = 0.95) -> dict:
    return _get("/stats/confidence/daily-cases", {"confidence": confidence})


def get_confidence_interval_daily_deaths(confidence: float = 0.95) -> dict:
    return _get("/stats/confidence/daily-deaths", {"confidence": confidence})


def get_forecast_state(state_code: str, days: int = 7) -> dict:
    return _get(f"/forecast/state/{state_code}", {"days": days})


def get_forecast_confidence(
    state_code: str, days: int = 7, confidence: float = 0.95
) -> dict:
    return _get(
        f"/forecast/state/{state_code}/confidence",
        {"days": days, "confidence": confidence},
    )


def get_forecast_municipality(
    state_code: str, municipality_code: int, days: int = 7
) -> dict:
    return _get(
        f"/forecast/municipality/{state_code}/{municipality_code}", {"days": days}
    )
