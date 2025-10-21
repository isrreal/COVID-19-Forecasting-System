from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_async_session
from src.models.casos_covid import CasoCovid
import logging
import pandas as pd
from scipy.stats import chi2_contingency
from unidecode import unidecode

logger = logging.getLogger(__name__)

async def get_summary_stats(session: AsyncSession):
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
            "total_records": row["total_records"],
            "total_confirmed": float(row["total_confirmed"] or 0),
            "total_deaths": float(row["total_deaths"] or 0),
            "avg_new_confirmed_per_day": float(row["avg_new_confirmed_per_day"] or 0),
            "avg_new_deaths_per_day": float(row["avg_new_deaths_per_day"] or 0),
        }
    except SQLAlchemyError as e:
        logger.error(f"Erro ao buscar estatísticas globais: {e}")
        return {"error": "Não foi possível obter as estatísticas globais."}


async def get_city_stats(city_name: str, state: str, session: AsyncSession):
    """
    Retorna estatísticas agregadas para uma cidade específica.
    """
    normalized_city_name = unidecode(city_name).lower()
    logger.info(f"Buscando estatísticas para a cidade: {normalized_city_name}")
    
    try:
        stmt = select(CasoCovid).where(CasoCovid.city != "N/A").limit(100)
        result = await session.execute(stmt)
        rows = result.scalars().all()  

        if not rows:  
            print("Nenhum registro encontrado")
        else:
            for row in rows:
                print(row.city, row.state, row.last_available_confirmed)
        result = await session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed"),
                func.sum(CasoCovid.last_available_deaths).label("total_deaths"),
                func.avg(CasoCovid.new_confirmed).label("avg_new_confirmed"),
                func.avg(CasoCovid.new_deaths).label("avg_new_deaths"),
            )
            .where(CasoCovid.city == normalized_city_name,
                    CasoCovid.state == state)
            .group_by(CasoCovid.city)
        )
        row = result.mappings().one_or_none()
        
        if row is None:
            return {"error": f"Nenhuma estatística encontrada para a cidade {city_name}."}
        
        return {
            "city": row["city"],
            "total_confirmed": float(row["total_confirmed"]) if row["total_confirmed"] else 0,
            "total_deaths": float(row["total_deaths"]) if row["total_deaths"] else 0,
            "avg_new_confirmed": float(row["avg_new_confirmed"]) if row["avg_new_confirmed"] else 0,
            "avg_new_deaths": float(row["avg_new_deaths"]) if row["avg_new_deaths"] else 0,
        }
    except SQLAlchemyError as e:
        logger.error(f"Erro ao buscar estatísticas para a cidade {city_name}: {e}")
        return {"error": f"Não foi possível obter as estatísticas para a cidade {city_name}."}


async def get_top_cities(limit: int, session: AsyncSession):
    """
    Retorna as cidades com mais casos confirmados acumulados.
    """
    try:
        result = await session.execute(
            select(
                CasoCovid.city,
                func.sum(CasoCovid.last_available_confirmed).label("total_confirmed")
            )
            .where(CasoCovid.city.isnot(None))
            .group_by(CasoCovid.city)
            .order_by(func.sum(CasoCovid.last_available_confirmed).desc())
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
        logger.error(f"Erro ao buscar top cidades: {e}")
        return {"error": "Não foi possível obter as cidades com mais casos confirmados."}


async def chi_square_state_deaths(session: AsyncSession):
    """
    Teste qui-quadrado entre estado e ocorrência de mortes.
    Retorna estatísticas do teste.
    """
    try:
        result = await session.execute(
            select(CasoCovid.state, CasoCovid.last_available_deaths)
            .where(CasoCovid.state.isnot(None))
        )
        rows = result.fetchall()
        
        df = pd.DataFrame(rows, columns=['state', 'deaths'])
        
        df['death_occurred'] = (df['deaths'] > 0).astype(int)
        
        contingency_table = pd.crosstab(df['state'], df['death_occurred'])
        
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        significance_level = 0.05
        result = "reject_null" if p < significance_level else "fail_to_reject_null"
        interpretation = (
            f"There is a statistically significant association between state and death occurrence (p < {significance_level})"
            if result == "reject_null"
            else f"No statistically significant association between state and death occurrence (p >= {significance_level})"
        )
        
        return {
            "test": "chi_square",
            "null_hypothesis": "Death occurrence is independent of state",
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "degrees_of_freedom": int(dof),
            "significance_level": significance_level,
            "result": result,
            "interpretation": interpretation,
            "contingency_table": contingency_table.to_dict(),
            "expected_frequencies": expected.tolist()
        }
        
    except SQLAlchemyError as e:
        logger.error(f"Erro ao realizar teste qui-quadrado: {e}")
        return {"error": "Não foi possível realizar o teste qui-quadrado."}
    except Exception as e:
        logger.error(f"Erro inesperado no teste qui-quadrado: {e}")
        return {"error": "Erro inesperado ao realizar o teste estatístico."}