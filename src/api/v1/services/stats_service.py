import logging
from typing import List, Dict, Union

import pandas as pd
from scipy.stats import chi2_contingency
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from unidecode import unidecode

from src.models.casos_covid import CasoCovid

logger = logging.getLogger(__name__)


# ==========================================================
# Estatísticas gerais
# ==========================================================
async def get_summary_stats(session: AsyncSession) -> Dict[str, Union[int, float, str]]:
    """
    Retorna estatísticas agregadas da base de casos de COVID-19.
    """
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

    except SQLAlchemyError as e:
        logger.exception("Erro ao buscar estatísticas globais.")
        return {"error": "Não foi possível obter as estatísticas globais."}


# ==========================================================
# Estatísticas por cidade
# ==========================================================
async def get_city_stats(city_name: str, state: str, session: AsyncSession) -> Dict[str, Union[str, float]]:
    """
    Retorna estatísticas agregadas para uma cidade e estado específicos.
    """
    normalized_city_name = unidecode(city_name).lower()
    logger.info(f"Buscando estatísticas para a cidade: {normalized_city_name}, estado: {state}")

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

    except SQLAlchemyError as e:
        logger.exception(f"Erro ao buscar estatísticas para a cidade {city_name}.")
        return {"error": f"Não foi possível obter as estatísticas para a cidade {city_name}."}


# ==========================================================
# Cidades com mais casos
# ==========================================================
async def get_top_cities(limit: int, session: AsyncSession) -> Union[List[Dict[str, Union[str, float]]], Dict[str, str]]:
    """
    Retorna as cidades com mais casos confirmados acumulados.
    """
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

        return [
            {
                "city": row["city"],
                "total_confirmed": float(row["total_confirmed"] or 0),
            }
            for row in rows
        ]

    except SQLAlchemyError:
        logger.exception("Erro ao buscar top cidades.")
        return {"error": "Não foi possível obter as cidades com mais casos confirmados."}


# ==========================================================
# Cidades mais letais
# ==========================================================
async def get_most_deadly_cities(limit: int, session: AsyncSession) -> Union[List[Dict[str, float]], Dict[str, str]]:
    """
    Retorna as cidades com o maior número acumulado de mortes.
    """
    try:
        result = await session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_deaths).label("total_deaths")
            )
            .where(CasoCovid.city.isnot(None), CasoCovid.city != "N/A")
            .group_by(CasoCovid.city)
            .order_by(func.sum(CasoCovid.last_available_deaths).desc())
            .limit(limit)
        )

        rows = result.mappings().all()

        return [
            {
                "city": row["city"],
                "total_deaths": float(row["total_deaths"] or 0),
            }
            for row in rows
        ]

    except SQLAlchemyError:
        logger.exception("Erro ao buscar cidades mais letais.")
        return {"error": "Não foi possível obter as cidades mais letais."}

# ==========================================================
# Cidades menos letais
# ==========================================================
async def get_least_affected_cities(limit: int, session: AsyncSession):
    """
    Retorna as cidades com menor número de casos confirmados acumulados.
    """
    try:
        result = await session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed")
            )
            .where(CasoCovid.city.isnot(None))
            .group_by(CasoCovid.city)
            .order_by(func.sum(CasoCovid.last_available_confirmed).asc())  
            .limit(limit)
        )

        rows = result.mappings().all()

        return [
            {
                "city": row["city"],
                "total_confirmed": float(row["total_confirmed"]) if row["total_confirmed"] else 0
            }
            for row in rows
        ]

    except SQLAlchemyError as e:
        logger.error(f"Erro ao buscar cidades menos afetadas: {e}")
        return {"error": "Não foi possível obter as cidades com menos casos confirmados."}

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
        reject = p < significance_level
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
