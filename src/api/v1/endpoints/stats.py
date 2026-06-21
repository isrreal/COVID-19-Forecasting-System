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
# Endpoints de Estatísticas (JSON)
# ==========================================================

@router.get("/summary", response_model = SummaryStats)
async def get_summary(db: AsyncSession = Depends(get_async_session)):
    """Retorna estatísticas gerais resumidas."""
    return await stats_service.get_summary_stats(db)


@router.get("/city/{city_name}/{state}", response_model = CityStats)
async def get_city(
    city_name: str = Path(min_length = 2, example = "Fortaleza"),
    state: str = Path(min_length = 2, max_length = 2, example = "CE"),
    db: AsyncSession = Depends(get_async_session)
):
    """Retorna estatísticas agregadas para uma cidade específica."""
    return await stats_service.get_city_stats(city_name, state, db)


@router.get("/top-cities", response_model = CityConfirmedList)
async def top_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Retorna as cidades com maior número de casos confirmados acumulados."""
    try:
        result = await stats_service.get_top_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar top cidades: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar top cidades: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro interno no servidor.")


@router.get("/chi-square/state-deaths", response_model = ChiSquareResult)
async def chi_square_test(db: AsyncSession = Depends(get_async_session)):
    """Realiza teste qui-quadrado entre estado e ocorrência de mortes."""
    return await stats_service.chi_square_state_deaths(db)


@router.get("/most-deadly-cities", response_model = CityMortalityList)
async def get_most_deadly_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Retorna as cidades com maior taxa de mortalidade (mortes / casos confirmados)."""
    try:
        result = await stats_service.get_most_deadly_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro interno no servidor.")


@router.get("/least-affected-cities", response_model = CityMortalityList)
async def get_least_affected_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """Retorna as cidades com menor número de casos confirmados acumulados."""
    try:
        result = await stats_service.get_least_affected_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar cidades menos afetadas: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar cidades menos afetadas: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = "Erro interno no servidor.")

# -----------------------------------------------------------
# Intervalos de confiança (JSON)
# -----------------------------------------------------------

@router.get("/confidence/cases", response_model = ConfidenceInterval)
async def confidence_interval_cases(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """Retorna o intervalo de confiança da média de novos casos diários."""
    try:
        return await stats_service.get_confidence_interval_cases(db, confidence)
    except Exception as e:
        logger.exception(f"Erro ao calcular IC de casos: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Erro ao calcular intervalo de confiança: {str(e)}")


@router.get("/confidence/deaths", response_model = ConfidenceInterval)
async def confidence_interval_deaths(
    confidence: float = Query(0.95, ge = 0.8, le = 0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """Retorna o intervalo de confiança da média de novas mortes diárias."""
    try:
        return await stats_service.get_confidence_interval_deaths(db, confidence)
    except Exception as e:
        logger.exception(f"Erro ao calcular IC de mortes: {e}")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"Erro ao calcular intervalo de confiança: {str(e)}")


@router.get("/plot/histogram", response_class = StreamingResponse)
async def histogram_plot(
    metric: str = Query("new_confirmed", description = "Métrica para o histograma"),
    bin_width: int = Query(100, description = "Largura dos intervalos (bins)"),
    max_value: int = Query(10000, description = "Valor máximo exibido no eixo X"),
    db: AsyncSession = Depends(get_async_session)
):
    """Endpoint que delega a geração do histograma para o serviço."""
    return await stats_service.generate_histogram(metric, bin_width, max_value, db)
