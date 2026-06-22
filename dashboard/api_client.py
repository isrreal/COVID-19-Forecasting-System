import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _get(path: str, params: dict = None) -> dict:
    response = requests.get(f"{API_BASE_URL}/api/v1{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_summary() -> dict:
    return _get("/stats/summary")


def get_top_cities(limit: int = 10) -> list:
    return _get("/stats/top-cities", {"limit": limit})["data"]


def get_most_deadly_cities(limit: int = 10) -> list:
    return _get("/stats/most-deadly-cities", {"limit": limit})["data"]


def get_least_affected_cities(limit: int = 10) -> list:
    return _get("/stats/least-affected-cities", {"limit": limit})["data"]


def get_confidence_interval_cases(confidence: float = 0.95) -> dict:
    return _get("/stats/confidence/cases", {"confidence": confidence})


def get_forecast_state(state_code: str, days: int = 7) -> dict:
    return _get(f"/forecast/state/{state_code}", {"days": days})


def get_forecast_confidence(state_code: str, days: int = 7, confidence: float = 0.95) -> dict:
    return _get(f"/forecast/state/{state_code}/confidence", {"days": days, "confidence": confidence})


def get_forecast_city(state_code: str, city_name: str, days: int = 7) -> dict:
    return _get(f"/forecast/city/{state_code}/{city_name}", {"days": days})
