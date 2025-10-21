from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_async_session, get_sync_session
from src.api.v1.services import stats_service

router: APIRouter = APIRouter()

@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_async_session)):
    return await stats_service.get_summary_stats(db)
     
@router.get("/city/{city_name}/{state}")
async def get_city(city_name: str, state: str, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna estatísticas agregadas para uma cidade específica.
    
    Args:
        city_name: Nome da cidade (case-insensitive, accent-insensitive)
        
    Returns:
        dict: Estatísticas da cidade (casos confirmados, mortes, médias)
    """
    return await stats_service.get_city_stats(city_name, state, db)

@router.get("/top-cities")
async def top_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    result = await stats_service.get_top_cities(limit, db)
    if "error" in result:
        raise HTTPException(status_code = 500, detail = result["error"])
    return result

@router.get("/chi-square/state-deaths")
async def chi_square_test(db: AsyncSession = Depends(get_async_session)):
    """
    Realiza teste qui-quadrado entre estado e ocorrência de mortes.
    
    Returns:
        dict: Estatísticas do teste (chi2, p-value, graus de liberdade, interpretação)
    """
    return await stats_service.chi_square_state_deaths(db)
