import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database import get_sync_session
from src.api.v1.schemas.stats import (
    ChiSquareResult,
    ConfidenceInterval,
    MunicipalityMortalityList,
    MunicipalityNotificationList,
    MunicipalityStats,
    SummaryStats,
)
from src.api.v1.services import stats_service

router: APIRouter = APIRouter()
logger = logging.getLogger(__name__)

# ==========================================================
# Statistics Endpoints (JSON)
# ==========================================================


@router.get("/summary", response_model=SummaryStats)
def get_summary(db: Session = Depends(get_sync_session)):
    """Returns aggregated summary statistics for the entire dengue dataset."""
    try:
        return stats_service.get_summary_stats(db)
    except Exception as e:
        logger.exception(f"Error fetching summary statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve summary statistics.",
        )


@router.get("/municipality/{municipality_code}", response_model=MunicipalityStats)
def get_municipality(
    municipality_code: int = Path(..., description="Municipality IBGE code"),
    db: Session = Depends(get_sync_session),
):
    """Returns aggregated statistics for a specific municipality."""
    try:
        result = stats_service.get_municipality_stats(municipality_code, db)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No statistics found for municipality {municipality_code}.",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Error fetching municipality statistics for {municipality_code}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve statistics for municipality {municipality_code}.",
        )


@router.get("/top-municipalities", response_model=MunicipalityNotificationList)
def top_municipalities(limit: int = 10, db: Session = Depends(get_sync_session)):
    """Returns municipalities with the highest notification counts."""
    try:
        result = stats_service.get_top_municipalities(limit, db)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        return {"data": result}
    except SQLAlchemyError as e:
        logger.exception(f"Database error while fetching top municipalities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database access error.",
        )
    except Exception as e:
        logger.exception(f"Unexpected error while fetching top municipalities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )


@router.get("/chi-square/state-deaths", response_model=ChiSquareResult)
def chi_square_test(db: Session = Depends(get_sync_session)):
    """Performs a chi-square test between state and death occurrence."""
    try:
        return stats_service.chi_square_state_deaths(db)
    except Exception as e:
        logger.exception(f"Error performing chi-square test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not perform the chi-square test.",
        )


@router.get("/most-deadly-municipalities", response_model=MunicipalityMortalityList)
def get_most_deadly_municipalities(
    limit: int = 10, db: Session = Depends(get_sync_session)
):
    """Returns municipalities with the highest mortality rate (deaths / notifications)."""
    try:
        result = stats_service.get_most_deadly_municipalities(limit, db)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        return {"data": result}
    except SQLAlchemyError as e:
        logger.exception(
            f"Database error while fetching most deadly municipalities: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database access error.",
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error while fetching most deadly municipalities: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )


@router.get("/least-affected-municipalities", response_model=MunicipalityMortalityList)
def get_least_affected_municipalities(
    limit: int = 10, db: Session = Depends(get_sync_session)
):
    """Returns municipalities with the lowest mortality rate."""
    try:
        result = stats_service.get_least_affected_municipalities(limit, db)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        return {"data": result}
    except SQLAlchemyError as e:
        logger.exception(
            f"Database error while fetching least affected municipalities: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database access error.",
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error while fetching least affected municipalities: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )


# -----------------------------------------------------------
# Confidence Intervals (JSON)
# -----------------------------------------------------------


@router.get("/confidence/daily-cases", response_model=ConfidenceInterval)
def confidence_interval_daily_cases(
    confidence: float = Query(0.95, ge=0.8, le=0.99),
    db: Session = Depends(get_sync_session),
):
    """Returns the confidence interval for the mean of daily dengue notifications."""
    try:
        return stats_service.get_confidence_interval_daily_cases(db, confidence)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for daily cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating confidence interval: {str(e)}",
        )


@router.get("/confidence/daily-deaths", response_model=ConfidenceInterval)
def confidence_interval_daily_deaths(
    confidence: float = Query(0.95, ge=0.8, le=0.99),
    db: Session = Depends(get_sync_session),
):
    """Returns the confidence interval for the mean of daily dengue deaths."""
    try:
        return stats_service.get_confidence_interval_daily_deaths(db, confidence)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Error calculating confidence interval for daily deaths: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating confidence interval: {str(e)}",
        )


# -----------------------------------------------------------
# Plots (PNG)
# -----------------------------------------------------------


@router.get("/plot/histogram", response_class=StreamingResponse)
def histogram_plot(
    metric: str = Query(
        "age_encoded",
        description="Metric to plot: age_encoded, final_classification, serotype, outcome",
    ),
    bin_width: int = Query(100, description="Width of histogram bins"),
    max_value: int = Query(10000, description="Maximum value displayed on the X axis"),
    db: Session = Depends(get_sync_session),
):
    """Returns a PNG histogram for the selected dengue metric."""
    return stats_service.generate_histogram(metric, bin_width, max_value, db)
