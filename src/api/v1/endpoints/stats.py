from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from src.database import get_async_session
from src.api.v1.services import stats_service
import logging

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_async_session)):
    """Retorna estatísticas gerais resumidas."""
    return await stats_service.get_summary_stats(db)


@router.get("/city/{city_name}/{state}")
async def get_city(city_name: str, state: str, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna estatísticas agregadas para uma cidade específica.
    
    Args:
        city_name: Nome da cidade (case-insensitive, accent-insensitive)
        state: Sigla do estado da cidade
    
    Returns:
        dict: Estatísticas da cidade (casos confirmados, mortes, médias)
    """
    return await stats_service.get_city_stats(city_name, state, db)


@router.get("/top-cities")
async def top_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna as cidades com maior número de casos confirmados acumulados.
    """
    try:
        result = await stats_service.get_top_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = 500, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar top cidades: {e}")
        raise HTTPException(status_code = 500, detail = "Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar top cidades: {e}")
        raise HTTPException(status_code = 500, detail = "Erro interno no servidor.")


@router.get("/chi-square/state-deaths")
async def chi_square_test(db: AsyncSession = Depends(get_async_session)):
    """
    Realiza teste qui-quadrado entre estado e ocorrência de mortes.
    
    Returns:
        dict: Estatísticas do teste (chi2, p-value, graus de liberdade, interpretação)
    """
    return await stats_service.chi_square_state_deaths(db)


@router.get("/most-deadly-cities")
async def get_most_deadly_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna as cidades com maior taxa de mortalidade (mortes / casos confirmados).
    """
    try:
        result = await stats_service.get_most_deadly_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = 500, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code = 500, detail = "Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code = 500, detail = "Erro interno no servidor.")
