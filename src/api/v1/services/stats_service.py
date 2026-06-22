import logging
from typing import Dict, Union
import io

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from unidecode import unidecode

from sqlalchemy import func, select, Column
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

from src.models.casos_covid import CasoCovid

logger = logging.getLogger(__name__)


# ==========================================================
# Summary Statistics
# ==========================================================
def get_summary_stats(session: Session) -> Dict[str, Union[int, float, str]]:
    result = session.execute(
        select(
            func.count(CasoCovid.id).label("total_records"),
            func.sum(CasoCovid.last_available_confirmed).label("total_confirmed"),
            func.sum(CasoCovid.last_available_deaths).label("total_deaths"),
            func.avg(CasoCovid.new_confirmed).label("avg_new_confirmed_per_day"),
            func.avg(CasoCovid.new_deaths).label("avg_new_deaths_per_day"),
        )
    )
    row = result.mappings().one()
    return {
        "total_records": int(row["total_records"] or 0),
        "total_confirmed": float(row["total_confirmed"] or 0),
        "total_deaths": float(row["total_deaths"] or 0),
        "avg_new_confirmed_per_day": float(row["avg_new_confirmed_per_day"] or 0),
        "avg_new_deaths_per_day": float(row["avg_new_deaths_per_day"] or 0),
    }


# ==========================================================
# City Statistics
# ==========================================================
def get_city_stats(
    city_name: str, state: str, session: Session
) -> Dict[str, Union[str, float]]:
    normalized_city_name = unidecode(city_name).lower()
    result = session.execute(
        select(
            CasoCovid.city,
            func.sum(CasoCovid.last_available_confirmed).label("total_confirmed"),
            func.sum(CasoCovid.last_available_deaths).label("total_deaths"),
            func.avg(CasoCovid.new_confirmed).label("avg_new_confirmed"),
            func.avg(CasoCovid.new_deaths).label("avg_new_deaths"),
        )
        .where(
            CasoCovid.city == normalized_city_name,
            CasoCovid.state == state,
            CasoCovid.city.isnot(None),
            CasoCovid.city != "N/A",
        )
        .group_by(CasoCovid.city)
    )
    row = result.mappings().one_or_none()
    if row is None:
        return None

    return {
        "city": row["city"],
        "total_confirmed": float(row["total_confirmed"] or 0),
        "total_deaths": float(row["total_deaths"] or 0),
        "avg_new_confirmed": float(row["avg_new_confirmed"] or 0),
        "avg_new_deaths": float(row["avg_new_deaths"] or 0),
    }


# ==========================================================
# Top / Most Deadly / Least Affected Cities
# ==========================================================
def get_top_cities(limit: int, session: Session):
    try:
        result = session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed"),
            )
            .where(CasoCovid.city.isnot(None), CasoCovid.city != "N/A")
            .group_by(CasoCovid.city)
            .order_by(func.sum(CasoCovid.last_available_confirmed).desc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {"city": r["city"], "total_confirmed": float(r["total_confirmed"] or 0)}
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching top cities.")
        return {"error": "Could not retrieve cities with the most confirmed cases."}


def get_most_deadly_cities(limit: int, session: Session):
    try:
        mortality_rate = (
            func.sum(CasoCovid.last_available_deaths)
            / func.sum(CasoCovid.last_available_confirmed)
        ).label("mortality_rate")
        result = session.execute(
            select(
                CasoCovid.city,
                CasoCovid.state,
                mortality_rate,
                func.sum(CasoCovid.last_available_deaths).label("total_deaths"),
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed"),
            )
            .where(CasoCovid.city.isnot(None), CasoCovid.city != "N/A")
            .group_by(CasoCovid.city, CasoCovid.state)
            .having(func.sum(CasoCovid.last_available_confirmed) > 0)
            .order_by(mortality_rate.desc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {
                "city": r["city"],
                "state": r["state"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": float(r["total_deaths"] or 0),
                "total_confirmed": float(r["total_confirmed"] or 0),
            }
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching most deadly cities.")
        return {"error": "Could not retrieve data."}


def get_least_affected_cities(limit: int, session: Session):
    try:
        total_confirmed_agg = func.sum(CasoCovid.last_available_confirmed)
        total_deaths_agg = func.sum(CasoCovid.last_available_deaths)
        mortality_rate = (total_deaths_agg / total_confirmed_agg).label(
            "mortality_rate"
        )
        result = session.execute(
            select(
                CasoCovid.city,
                CasoCovid.state,
                mortality_rate,
                total_deaths_agg.label("total_deaths"),
                total_confirmed_agg.label("total_confirmed"),
            )
            .where(CasoCovid.city.isnot(None), CasoCovid.city != "N/A")
            .group_by(CasoCovid.city, CasoCovid.state)
            .having(total_confirmed_agg > 0)
            .order_by(mortality_rate.asc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {
                "city": r["city"],
                "state": r["state"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": float(r["total_deaths"] or 0),
                "total_confirmed": float(r["total_confirmed"] or 0),
            }
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching least affected cities.")
        return {"error": "Could not retrieve data."}


# ==========================================================
# Chi-Square Test
# ==========================================================
def chi_square_state_deaths(session: Session) -> Dict[str, Union[str, float, Dict]]:
    """Performs a chi-square test for association between state and death occurrence."""
    try:
        result = session.execute(
            select(CasoCovid.state, CasoCovid.last_available_deaths).where(
                CasoCovid.state.isnot(None)
            )
        )
        rows = result.fetchall()

        if not rows:
            return {"error": "No data available to perform the chi-square test."}

        df = pd.DataFrame(rows, columns=["state", "deaths"])
        df["death_occurred"] = (df["deaths"] > 0).astype(int)

        contingency = pd.crosstab(df["state"], df["death_occurred"])
        chi2, p, dof, expected = chi2_contingency(contingency)

        significance_level = 0.05
        reject = bool(p < significance_level)

        interpretation = (
            f"There is a statistically significant association between state and death occurrence (p < {significance_level})"
            if reject
            else f"There is no statistically significant association between state and death occurrence (p ≥ {significance_level})"
        )

        return {
            "test": "chi_square",
            "null_hypothesis": "Death occurrence is independent of the state",
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "degrees_of_freedom": int(dof),
            "significance_level": significance_level,
            "reject_null_hypothesis": reject,
            "interpretation": interpretation,
            "contingency_table": contingency.to_dict(),
            "expected_frequencies": expected.tolist(),
        }

    except SQLAlchemyError:
        logger.exception("SQL error while performing chi-square test.")
        return {
            "error": "Could not perform the chi-square test due to a database error."
        }
    except Exception:
        logger.exception("Unexpected error while performing chi-square test.")
        return {"error": "Unexpected error while performing the statistical test."}


# ==========================================================
# Confidence Intervals
# ==========================================================
def _get_confidence_interval(
    session: Session, metric_col: Column, metric_name: str, confidence=0.95
):
    result = session.execute(
        select(
            func.avg(metric_col).label("mean"),
            func.stddev(metric_col).label("stddev"),
            func.count(metric_col).label("n"),
        ).where(metric_col.isnot(None))
    )
    row = result.mappings().one_or_none()
    if not row or row["n"] < 2:
        raise ValueError(
            f"Insufficient data to compute confidence interval for '{metric_name}'."
        )

    mean = float(row["mean"])
    stddev = float(row["stddev"])
    n = int(row["n"])
    sem = stddev / np.sqrt(n)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return {
        "metric": metric_name,
        "mean": mean,
        "lower": mean - h,
        "upper": mean + h,
        "n": n,
    }


def get_confidence_interval_cases(session: Session, confidence=0.95):
    return _get_confidence_interval(
        session, CasoCovid.new_confirmed, "new_confirmed", confidence
    )


def get_confidence_interval_deaths(session: Session, confidence=0.95):
    return _get_confidence_interval(
        session, CasoCovid.new_deaths, "new_deaths", confidence
    )


def generate_histogram(metric: str, bin_width: int, max_value: int, session: Session):
    """Generates a histogram for the selected metric and returns a PNG image."""

    metric_map = {
        "new_confirmed": CasoCovid.new_confirmed,
        "new_deaths": CasoCovid.new_deaths,
        "last_available_confirmed": CasoCovid.last_available_confirmed,
        "last_available_deaths": CasoCovid.last_available_deaths,
    }

    if metric not in metric_map:
        return {"error": f"Invalid metric: {metric}"}

    try:
        result = session.execute(
            select(metric_map[metric]).where(metric_map[metric].isnot(None))
        )
        values = [row[0] for row in result.fetchall() if row[0] is not None]

        if not values:
            return {"error": "No data available to generate the histogram."}

        df = pd.DataFrame(values, columns=[metric])
        df = df[df[metric] <= max_value]

        plt.figure(figsize=(8, 5))
        plt.hist(
            df[metric],
            bins=np.arange(0, max_value + bin_width, bin_width),
            edgecolor="black",
        )
        plt.title(f"Histogram of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except SQLAlchemyError:
        logger.exception("Database error while generating histogram.")
        return {"error": "Database error while generating the histogram."}
    except Exception:
        logger.exception("Unexpected error while generating histogram.")
        return {"error": "Unexpected error while generating the histogram."}
