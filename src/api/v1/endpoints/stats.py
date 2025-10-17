from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_session
from src.api.v1.services import stats_service

router: APIRouter = APIRouter()

@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_session)):
    """
    Retorna estatísticas globais do dataset COVID-19.
    
    Returns:
        dict: Estatísticas agregadas (total de registros, casos confirmados, mortes, médias)
    """
    result = await stats_service.get_summary_stats()
    if "error" in result:
        raise HTTPException(status_code = 500, detail = result["error"])
    return result


@router.get("/city/{city_name}/{state}")
async def get_city(city_name: str, state: str, db: AsyncSession = Depends(get_session)):
    """
    Retorna estatísticas agregadas para uma cidade específica.
    
    Args:
        city_name: Nome da cidade (case-insensitive, accent-insensitive)
        
    Returns:
        dict: Estatísticas da cidade (casos confirmados, mortes, médias)
    """
    result = await stats_service.get_city_stats(city_name, state)
    if "error" in result:
        raise HTTPException(status_code = 404, detail = result["error"])
    return result


@router.get("/top-cities")
async def get_top_cities(limit: int = 10, db: AsyncSession = Depends(get_session)):
    """
    Retorna as cidades com mais casos confirmados acumulados.
    
    Args:
        limit: Número de cidades a retornar (padrão: 10)
        
    Returns:
        list: Lista de cidades ordenadas por casos confirmados
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code = 400, detail = "Limit must be between 1 and 100")
    
    result = await stats_service.get_top_cities(limit)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code = 500, detail = result["error"])
    return result


@router.get("/chi-square/state-deaths")
async def chi_square_test(db: AsyncSession = Depends(get_session)):
    """
    Realiza teste qui-quadrado entre estado e ocorrência de mortes.
    
    Returns:
        dict: Estatísticas do teste (chi2, p-value, graus de liberdade, interpretação)
    """
    result = await stats_service.chi_square_state_deaths()
    if "error" in result:
        raise HTTPException(status_code = 500, detail = result["error"])
    return result