import logging
from typing import List, Dict, Union, Any
import io

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from unidecode import unidecode

from sqlalchemy import func, select, case, Column
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool

from src.models.casos_covid import CasoCovid

logger = logging.getLogger(__name__)


# ==========================================================
# Estatísticas gerais
# ==========================================================
async def get_summary_stats(session: AsyncSession) -> Dict[str, Union[int, float, str]]:
    try:
        result = await session.execute(
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
    except SQLAlchemyError:
        logger.exception("Erro ao buscar estatísticas globais.")
        return {"error": "Não foi possível obter as estatísticas globais."}


# ==========================================================
# Estatísticas por cidade
# ==========================================================
async def get_city_stats(city_name: str, state: str, session: AsyncSession) -> Dict[str, Union[str, float]]:
    normalized_city_name = unidecode(city_name).lower()
    try:
        result = await session.execute(
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
                CasoCovid.city != "N/A"
            )
            .group_by(CasoCovid.city)
        )
        row = result.mappings().one_or_none()
        if row is None:
            return {"error": f"Nenhuma estatística encontrada para {city_name} - {state}."}

        return {
            "city": row["city"],
            "total_confirmed": float(row["total_confirmed"] or 0),
            "total_deaths": float(row["total_deaths"] or 0),
            "avg_new_confirmed": float(row["avg_new_confirmed"] or 0),
            "avg_new_deaths": float(row["avg_new_deaths"] or 0),
        }
    except SQLAlchemyError:
        logger.exception(f"Erro ao buscar estatísticas para a cidade {city_name}.")
        return {"error": f"Não foi possível obter as estatísticas para a cidade {city_name}."}


# ==========================================================
# Cidades com mais casos / mais letais / menos letais
# ==========================================================
async def get_top_cities(limit: int, session: AsyncSession):
    try:
        result = await session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed")
            )
            .where(CasoCovid.city.isnot(None), CasoCovid.city != "N/A")
            .group_by(CasoCovid.city)
            .order_by(func.sum(CasoCovid.last_available_confirmed).desc())
            .limit(limit)
        )
        rows = result.mappings().all()
        return [{"city": r["city"], "total_confirmed": float(r["total_confirmed"] or 0)} for r in rows]
    except SQLAlchemyError:
        logger.exception("Erro ao buscar top cidades.")
        return {"error": "Não foi possível obter as cidades com mais casos confirmados."}


async def get_most_deadly_cities(limit: int, session: AsyncSession):
    try:
        mortality_rate = (func.sum(CasoCovid.last_available_deaths) /
                          func.sum(CasoCovid.last_available_confirmed)).label("mortality_rate")
        result = await session.execute(
            select(
                CasoCovid.city,
                CasoCovid.state,
                mortality_rate,
                func.sum(CasoCovid.last_available_deaths).label("total_deaths"),
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed")
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
                "city": r["city"], "state": r["state"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": float(r["total_deaths"] or 0),
                "total_confirmed": float(r["total_confirmed"] or 0)
            } for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Erro ao buscar cidades mais letais.")
        return {"error": "Não foi possível obter os dados."}


async def get_least_affected_cities(limit: int, session: AsyncSession):
    try:
        total_confirmed_agg = func.sum(CasoCovid.last_available_confirmed)
        total_deaths_agg = func.sum(CasoCovid.last_available_deaths)
        mortality_rate = (total_deaths_agg / total_confirmed_agg).label("mortality_rate")
        result = await session.execute(
            select(
                CasoCovid.city,
                CasoCovid.state,
                mortality_rate,
                total_deaths_agg.label("total_deaths"),
                total_confirmed_agg.label("total_confirmed")
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
                "city": r["city"], "state": r["state"],
                "mortality_rate": float(r["mortality_rate"] or 0),
                "total_deaths": float(r["total_deaths"] or 0),
                "total_confirmed": float(r["total_confirmed"] or 0)
            } for r in rows
        ]
    except SQLAlchemyError:
        logger.exception("Erro ao buscar cidades menos letais.")
        return {"error": "Não foi possível obter os dados."}


# ==========================================================
# Teste Qui-quadrado entre estado e ocorrência de mortes
# ==========================================================
async def chi_square_state_deaths(session: AsyncSession) -> Dict[str, Union[str, float, Dict]]:
    """
    Realiza teste qui-quadrado para verificar associação entre estado e ocorrência de mortes.
    """
    try:
        result = await session.execute(
            select(CasoCovid.state, CasoCovid.last_available_deaths)
            .where(CasoCovid.state.isnot(None))
        )
        rows = result.fetchall()

        if not rows:
            return {"error": "Nenhum dado disponível para realizar o teste qui-quadrado."}

        df = pd.DataFrame(rows, columns=["state", "deaths"])
        df["death_occurred"] = (df["deaths"] > 0).astype(int)

        contingency = pd.crosstab(df["state"], df["death_occurred"])
        chi2, p, dof, expected = chi2_contingency(contingency)

        significance_level = 0.05
        reject = bool(p < significance_level)  

        interpretation = (
            f"Existe associação estatisticamente significativa entre estado e ocorrência de mortes (p < {significance_level})"
            if reject else
            f"Não há associação estatisticamente significativa entre estado e ocorrência de mortes (p ≥ {significance_level})"
        )

        return {
            "test": "chi_square",
            "null_hypothesis": "A ocorrência de mortes é independente do estado",
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
        logger.exception("Erro SQL ao realizar teste qui-quadrado.")
        return {"error": "Não foi possível realizar o teste qui-quadrado devido a erro de banco de dados."}
    except Exception:
        logger.exception("Erro inesperado ao realizar o teste qui-quadrado.")
        return {"error": "Erro inesperado ao realizar o teste estatístico."}


# ==========================================================
# Intervalos de Confiança
# ==========================================================
async def _get_confidence_interval(session: AsyncSession, metric_col: Column, metric_name: str, confidence = 0.95):
    try:
        result = await session.execute(
            select(
                func.avg(metric_col).label("mean"),
                func.stddev(metric_col).label("stddev"),
                func.count(metric_col).label("n")
            ).where(metric_col.isnot(None))
        )
        row = result.mappings().one_or_none()
        if not row or row["n"] < 2:
            return {"error": "Dados insuficientes"}

        mean = float(row["mean"])
        stddev = float(row["stddev"])
        n = int(row["n"])
        sem = stddev / np.sqrt(n)
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        return {
            "metric": metric_name,
            "mean": mean,
            "lower": mean - h,
            "upper": mean + h,
            "n": n
        }
    except Exception:
        logger.exception(f"Erro no IC para {metric_name}.")
        return {"error": "Erro ao calcular IC."}


async def get_confidence_interval_cases(session, confidence = 0.95):
    return await _get_confidence_interval(session, CasoCovid.new_confirmed, "new_confirmed", confidence)


async def get_confidence_interval_deaths(session, confidence = 0.95):
    return await _get_confidence_interval(session, CasoCovid.new_deaths, "new_deaths", confidence)


# ==========================================================
# Helper
# ==========================================================
def _get_metric_column(metric: str):
    metric_map = {
        "new_confirmed": CasoCovid.new_confirmed,
        "new_deaths": CasoCovid.new_deaths,
        "last_available_confirmed": CasoCovid.last_available_confirmed,
        "last_available_deaths": CasoCovid.last_available_deaths,
    }
    col = metric_map.get(metric)
    if col is None:
        raise ValueError("Métrica inválida.")
    return col


# ==========================================================
# Dados para gráficos + Imagens
# ==========================================================
async def get_histogram_data(session: AsyncSession, metric = "new_confirmed", bin_width = 100, max_value = 10000):
    try:
        col = _get_metric_column(metric)
        bin_start = (func.floor(col / bin_width) * bin_width).label("bin")
        result = await session.execute(
            select(bin_start, func.count().label("count"))
            .where(col.isnot(None), col < max_value)
            .group_by(bin_start)
            .order_by(bin_start)
        )
        rows = result.mappings().all()
        return {"metric": metric, "bin_width": bin_width, "histogram_data": rows}
    except Exception:
        return {"error": "Erro ao obter histograma."}


def _plot_histogram(data):
    df = pd.DataFrame(data["histogram_data"])
    if df.empty:
        return io.BytesIO()
    fig, ax = plt.subplots()
    ax.bar(df["bin"], df["count"], width=data["bin_width"])
    ax.set_title("Histograma")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


async def get_histogram_image(session, metric="new_confirmed", bin_width = 100, max_value = 10000):
    data = await get_histogram_data(session, metric, bin_width, max_value)
    if "error" in data:
        return data
    return await run_in_threadpool(_plot_histogram, data)


async def get_time_series_data(session, metric = "new_confirmed", state = None):
    try:
        col = _get_metric_column(metric)
        query = (
            select(CasoCovid.datetime, func.sum(col))
            .where(col.isnot(None))
            .group_by(CasoCovid.datetime)
            .order_by(CasoCovid.datetime)
        )
        if state:
            query = query.where(CasoCovid.state == state)
        rows = (await session.execute(query)).mappings().all()
        return {"time_series_data": rows, "metric": metric}
    except Exception:
        return {"error": "Erro ao obter série temporal."}


def _plot_time_series(data):
    df = pd.DataFrame(data["time_series_data"])
    if df.empty:
        return io.BytesIO()
    fig, ax = plt.subplots()
    ax.plot(df["datetime"], df["sum_1"])
    ax.set_title("Série temporal")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


async def get_time_series_image(session, metric = "new_confirmed", state = None):
    data = await get_time_series_data(session, metric, state)
    if "error" in data:
        return data
    return await run_in_threadpool(_plot_time_series, data)
