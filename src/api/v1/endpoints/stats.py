import logging
from fastapi import APIRouter, Depends, Query, HTTPException, Path, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from database import get_async_session
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
async def get_summary(db: AsyncSession = Depends(get_async_session)):
    """Returns aggregated summary statistics."""
    return await stats_service.get_summary_stats(db)


@router.get("/city/{city_name}/{state}", response_model = CityStats)
async def get_city(
    city_name: str = Path(min_length = 2, example = "Fortaleza"),
    state: str = Path(min_length = 2, max_length = 2, example = "CE"),
    db: AsyncSession = Depends(get_async_session)
):
    """Returns aggregated statistics for a specific city."""
    return await stats_service.get_city_stats(city_name, state, db)


@router.get("/top-cities", response_model = CityConfirmedList)
async def top_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Returns cities with the highest cumulative confirmed case counts."""
    try:
        result = await stats_service.get_top_cities(limit, db)

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
async def chi_square_test(db: AsyncSession = Depends(get_async_session)):
    """Performs a chi-square test between state and death occurrence."""
    return await stats_service.chi_square_state_deaths(db)


@router.get("/most-deadly-cities", response_model = CityMortalityList)
async def get_most_deadly_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Returns cities with the highest mortality rate (deaths / confirmed cases)."""
    try:
        result = await stats_service.get_most_deadly_cities(limit, db)

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
async def get_least_affected_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Returns cities with the lowest cumulative confirmed case counts."""
    try:
        result = await stats_service.get_least_affected_cities(limit, db)

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
async def confidence_interval_cases(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """Returns the confidence interval for the mean of daily new confirmed cases."""
    try:
        return await stats_service.get_confidence_interval_cases(db, confidence)
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for cases: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Error calculating confidence interval: {str(e)}")


@router.get("/confidence/deaths", response_model = ConfidenceInterval)
async def confidence_interval_deaths(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """Returns the confidence interval for the mean of daily new deaths."""
    try:
        return await stats_service.get_confidence_interval_deaths(db, confidence)
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for deaths: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Error calculating confidence interval: {str(e)}")


@router.get("/plot/histogram", response_class = StreamingResponse)
async def histogram_plot(
    metric: str = Query("new_confirmed", description = "Metric to plot in the histogram"),
    bin_width: int = Query(100, description = "Width of histogram bins"),
    max_value: int = Query(10000, description = "Maximum value displayed on the X axis"),
    db: AsyncSession = Depends(get_async_session)
):
    """Returns a PNG histogram for the selected metric."""
    return await stats_service.generate_histogram(metric, bin_width, max_value, db)
