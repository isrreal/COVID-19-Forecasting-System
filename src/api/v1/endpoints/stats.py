import logging
from fastapi import APIRouter, Depends, Query, HTTPException, Path, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import get_sync_session
from src.api.v1.services import stats_service
from src.api.v1.schemas.stats import (
    SummaryStats,
    CityStats,
    CityConfirmedList,
    CityMortalityList,
    ChiSquareResult,
    ConfidenceInterval,
)

router: APIRouter = APIRouter()
logger = logging.getLogger(__name__)

# ==========================================================
# Statistics Endpoints (JSON)
# ==========================================================

@router.get("/summary", response_model = SummaryStats)
def get_summary(db: Session = Depends(get_sync_session)):
    """Returns aggregated summary statistics."""
    try:
        return stats_service.get_summary_stats(db)
    except Exception as e:
        logger.exception(f"Error fetching summary statistics: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Could not retrieve summary statistics.")


@router.get("/city/{city_name}/{state}", response_model = CityStats)
def get_city(
    city_name: str = Path(min_length = 2, examples={"default": {"value": "Fortaleza"}}),
    state: str = Path(min_length = 2, max_length = 2, examples={"default": {"value": "CE"}}),
    db: Session = Depends(get_sync_session)
):
    """Returns aggregated statistics for a specific city."""
    try:
        result = stats_service.get_city_stats(city_name, state, db)
        if result is None:
            raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f"No statistics found for {city_name} - {state}.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching city statistics for {city_name}: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Could not retrieve statistics for city {city_name}.")


@router.get("/top-cities", response_model = CityConfirmedList)
def top_cities(limit: int = 10, db: Session = Depends(get_sync_session)):
    """Returns cities with the highest cumulative confirmed case counts."""
    try:
        result = stats_service.get_top_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Database error while fetching top cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Database access error.")

    except Exception as e:
        logger.exception(f"Unexpected error while fetching top cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Internal server error.")


@router.get("/chi-square/state-deaths", response_model = ChiSquareResult)
def chi_square_test(db: Session = Depends(get_sync_session)):
    """Performs a chi-square test between state and death occurrence."""
    try:
        return stats_service.chi_square_state_deaths(db)
    except Exception as e:
        logger.exception(f"Error performing chi-square test: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Could not perform the chi-square test.")


@router.get("/most-deadly-cities", response_model = CityMortalityList)
def get_most_deadly_cities(limit: int = 10, db: Session = Depends(get_sync_session)):
    """Returns cities with the highest mortality rate (deaths / confirmed cases)."""
    try:
        result = stats_service.get_most_deadly_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Database error while fetching most deadly cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Database access error.")

    except Exception as e:
        logger.exception(f"Unexpected error while fetching most deadly cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Internal server error.")


@router.get("/least-affected-cities", response_model = CityMortalityList)
def get_least_affected_cities(limit: int = 10, db: Session = Depends(get_sync_session)):
    """Returns cities with the lowest cumulative confirmed case counts."""
    try:
        result = stats_service.get_least_affected_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Database error while fetching least affected cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Database access error.")

    except Exception as e:
        logger.exception(f"Unexpected error while fetching least affected cities: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Internal server error.")

# -----------------------------------------------------------
# Confidence Intervals (JSON)
# -----------------------------------------------------------

@router.get("/confidence/cases", response_model = ConfidenceInterval)
def confidence_interval_cases(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: Session = Depends(get_sync_session)
):
    """Returns the confidence interval for the mean of daily new confirmed cases."""
    try:
        return stats_service.get_confidence_interval_cases(db, confidence)
    except ValueError as e:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = str(e))
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for cases: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Error calculating confidence interval: {str(e)}")


@router.get("/confidence/deaths", response_model = ConfidenceInterval)
def confidence_interval_deaths(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: Session = Depends(get_sync_session)
):
    """Returns the confidence interval for the mean of daily new deaths."""
    try:
        return stats_service.get_confidence_interval_deaths(db, confidence)
    except ValueError as e:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = str(e))
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for deaths: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Error calculating confidence interval: {str(e)}")


@router.get("/plot/histogram", response_class = StreamingResponse)
def histogram_plot(
    metric: str = Query("new_confirmed", description = "Metric to plot in the histogram"),
    bin_width: int = Query(100, description = "Width of histogram bins"),
    max_value: int = Query(10000, description = "Maximum value displayed on the X axis"),
    db: Session = Depends(get_sync_session)
):
    """Returns a PNG histogram for the selected metric."""
    return stats_service.generate_histogram(metric, bin_width, max_value, db)
