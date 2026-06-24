import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency

from sqlalchemy import case, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

from src.models.caso_dengue import CasoDengue

logger = logging.getLogger(__name__)

_OUTCOME_DEATH = 2
_HOSPITALIZED_YES = 1


# ==========================================================
# Summary Statistics
# ==========================================================


def get_summary_stats(session: Session) -> dict[str, int | float]:
    total = session.execute(select(func.count(CasoDengue.id))).scalar() or 0
    deaths = (
        session.execute(
            select(func.count(CasoDengue.id)).where(
                CasoDengue.outcome == _OUTCOME_DEATH
            )
        ).scalar()
        or 0
    )
    hospitalized = (
        session.execute(
            select(func.count(CasoDengue.id)).where(
                CasoDengue.hospitalized == _HOSPITALIZED_YES
            )
        ).scalar()
        or 0
    )

    return {
        "total_notifications": int(total),
        "total_deaths": int(deaths),
        "hospitalization_rate": round(hospitalized / total, 4) if total else 0.0,
        "mortality_rate": round(deaths / total, 4) if total else 0.0,
    }


# ==========================================================
# Municipality Statistics
# ==========================================================


def get_municipality_stats(
    municipality_code: int, session: Session
) -> dict[str, int | float] | None:
    result = session.execute(
        select(
            func.count(CasoDengue.id).label("total_notifications"),
            func.sum(case((CasoDengue.outcome == _OUTCOME_DEATH, 1), else_=0)).label(
                "total_deaths"
            ),
            func.sum(
                case((CasoDengue.hospitalized == _HOSPITALIZED_YES, 1), else_=0)
            ).label("total_hospitalized"),
        ).where(CasoDengue.municipality_ibge_code == municipality_code)
    )
    row = result.mappings().one_or_none()
    if row is None or row["total_notifications"] == 0:
        return None

    total = int(row["total_notifications"])
    deaths = int(row["total_deaths"] or 0)
    hospitalized = int(row["total_hospitalized"] or 0)

    return {
        "municipality_code": municipality_code,
        "total_notifications": total,
        "total_deaths": deaths,
        "hospitalization_rate": round(hospitalized / total, 4),
        "mortality_rate": round(deaths / total, 4),
    }


# ==========================================================
# Top / Most Deadly / Least Affected Municipalities
# ==========================================================


def get_top_municipalities(limit: int, session: Session):
    try:
        result = session.execute(
            select(
                CasoDengue.municipality_ibge_code,
                CasoDengue.state_ibge_code,
                func.count(CasoDengue.id).label("total_notifications"),
            )
            .group_by(CasoDengue.municipality_ibge_code, CasoDengue.state_ibge_code)
            .order_by(func.count(CasoDengue.id).desc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {
                "municipality_code": r["municipality_ibge_code"],
                "state_code": r["state_ibge_code"],
                "total_notifications": int(r["total_notifications"]),
            }
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching top municipalities.")
        return {
            "error": "Could not retrieve municipalities with the most notifications."
        }


def get_most_deadly_municipalities(limit: int, session: Session):
    try:
        total_agg = func.count(CasoDengue.id)
        deaths_agg = func.sum(case((CasoDengue.outcome == _OUTCOME_DEATH, 1), else_=0))
        mortality_rate = (deaths_agg / total_agg).label("mortality_rate")

        result = session.execute(
            select(
                CasoDengue.municipality_ibge_code,
                CasoDengue.state_ibge_code,
                mortality_rate,
                deaths_agg.label("total_deaths"),
                total_agg.label("total_notifications"),
            )
            .group_by(CasoDengue.municipality_ibge_code, CasoDengue.state_ibge_code)
            .having(total_agg > 0)
            .order_by(mortality_rate.desc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {
                "municipality_code": r["municipality_ibge_code"],
                "state_code": r["state_ibge_code"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": int(r["total_deaths"] or 0),
                "total_notifications": int(r["total_notifications"] or 0),
            }
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching most deadly municipalities.")
        return {"error": "Could not retrieve data."}


def get_least_affected_municipalities(limit: int, session: Session):
    try:
        total_agg = func.count(CasoDengue.id)
        deaths_agg = func.sum(case((CasoDengue.outcome == _OUTCOME_DEATH, 1), else_=0))
        mortality_rate = (deaths_agg / total_agg).label("mortality_rate")

        result = session.execute(
            select(
                CasoDengue.municipality_ibge_code,
                CasoDengue.state_ibge_code,
                mortality_rate,
                deaths_agg.label("total_deaths"),
                total_agg.label("total_notifications"),
            )
            .group_by(CasoDengue.municipality_ibge_code, CasoDengue.state_ibge_code)
            .having(total_agg > 0)
            .order_by(mortality_rate.asc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [
            {
                "municipality_code": r["municipality_ibge_code"],
                "state_code": r["state_ibge_code"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": int(r["total_deaths"] or 0),
                "total_notifications": int(r["total_notifications"] or 0),
            }
            for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Error fetching least affected municipalities.")
        return {"error": "Could not retrieve data."}


# ==========================================================
# Chi-Square Test
# ==========================================================


def chi_square_state_deaths(session: Session) -> dict[str, str | float | dict]:
    """Performs a chi-square test for association between state and death occurrence."""
    try:
        result = session.execute(
            select(CasoDengue.state_ibge_code, CasoDengue.outcome).where(
                CasoDengue.state_ibge_code.isnot(None)
            )
        )
        rows = result.fetchall()

        if not rows:
            return {"error": "No data available to perform the chi-square test."}

        df = pd.DataFrame(rows, columns=["state_ibge_code", "outcome"])
        df["death_occurred"] = (df["outcome"] == _OUTCOME_DEATH).astype(int)

        contingency = pd.crosstab(df["state_ibge_code"], df["death_occurred"])
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


def get_confidence_interval_daily_cases(
    session: Session, confidence: float = 0.95
) -> dict:
    """Computes a confidence interval for the daily notification count."""
    try:
        subq = (
            select(func.count(CasoDengue.id).label("daily_count"))
            .group_by(CasoDengue.notification_date)
            .subquery()
        )
        result = session.execute(
            select(
                func.avg(subq.c.daily_count).label("mean"),
                func.stddev(subq.c.daily_count).label("stddev"),
                func.count(subq.c.daily_count).label("n"),
            )
        )
        row = result.mappings().one_or_none()
        if not row or row["n"] < 2:
            raise ValueError("Insufficient data to compute confidence interval.")

        mean = float(row["mean"])
        stddev = float(row["stddev"])
        n = int(row["n"])
        sem = stddev / np.sqrt(n)
        h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)

        return {
            "metric": "daily_notifications",
            "mean": mean,
            "lower": mean - h,
            "upper": mean + h,
            "n": n,
        }

    except SQLAlchemyError:
        logger.exception(
            "Database error computing confidence interval for daily cases."
        )
        return {"error": "Database error computing confidence interval."}


def get_confidence_interval_daily_deaths(
    session: Session, confidence: float = 0.95
) -> dict:
    """Computes a confidence interval for the daily death count."""
    try:
        subq = (
            select(
                func.sum(
                    case((CasoDengue.outcome == _OUTCOME_DEATH, 1), else_=0)
                ).label("daily_deaths")
            )
            .group_by(CasoDengue.notification_date)
            .subquery()
        )
        result = session.execute(
            select(
                func.avg(subq.c.daily_deaths).label("mean"),
                func.stddev(subq.c.daily_deaths).label("stddev"),
                func.count(subq.c.daily_deaths).label("n"),
            )
        )
        row = result.mappings().one_or_none()
        if not row or row["n"] < 2:
            raise ValueError("Insufficient data to compute confidence interval.")

        mean = float(row["mean"])
        stddev = float(row["stddev"])
        n = int(row["n"])
        sem = stddev / np.sqrt(n)
        h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)

        return {
            "metric": "daily_deaths",
            "mean": mean,
            "lower": mean - h,
            "upper": mean + h,
            "n": n,
        }

    except SQLAlchemyError:
        logger.exception(
            "Database error computing confidence interval for daily deaths."
        )
        return {"error": "Database error computing confidence interval."}


# ==========================================================
# Histogram
# ==========================================================


def generate_histogram(metric: str, bin_width: int, max_value: int, session: Session):
    """Generates a histogram for the selected dengue metric and returns a PNG image."""
    metric_map = {
        "age_encoded": CasoDengue.age_encoded,
        "final_classification": CasoDengue.final_classification,
        "serotype": CasoDengue.serotype,
        "outcome": CasoDengue.outcome,
    }

    if metric not in metric_map:
        return {
            "error": f"Invalid metric: '{metric}'. Valid options: {list(metric_map.keys())}"
        }

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
