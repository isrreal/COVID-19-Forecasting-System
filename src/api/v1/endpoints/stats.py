import logging  # <-- CORREÇÃO: Importado
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from database import get_async_session  # Supondo que este é o caminho correto
from src.api.v1.services import stats_service
import io

router: APIRouter = APIRouter()
logger = logging.getLogger(__name__)  # <-- CORREÇÃO: Logger definido

# ==========================================================
# Endpoints de Estatísticas (JSON)
# ==========================================================

@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_async_session)):
    """Retorna estatísticas gerais resumidas."""
    # O try/except é tratado no serviço ou é simples o suficiente
    return await stats_service.get_summary_stats(db)


@router.get("/city/{city_name}/{state}")
async def get_city(city_name: str, state: str, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna estatísticas agregadas para uma cidade específica.
    """
    # O try/except é tratado no serviço ou é simples o suficiente
    return await stats_service.get_city_stats(city_name, state, db)


@router.get("/top-cities")
async def top_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna as cidades com maior número de casos confirmados acumulados.
    """
    try:
        result = await stats_service.get_top_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar top cidades: {e}")
        raise HTTPException(status_code=500, detail="Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar top cidades: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")


@router.get("/chi-square/state-deaths")
async def chi_square_test(db: AsyncSession = Depends(get_async_session)):
    """
    Realiza teste qui-quadrado entre estado e ocorrência de mortes.
    """
    # O try/except foi movido para dentro do serviço
    return await stats_service.chi_square_state_deaths(db)


@router.get("/most-deadly-cities")
async def get_most_deadly_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna as cidades com maior taxa de mortalidade (mortes / casos confirmados).
    """
    try:
        result = await stats_service.get_most_deadly_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code=500, detail="Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar cidades mais letais: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")


@router.get("/least-affected-cities")
async def get_least_affected_cities(limit: int = 10, db: AsyncSession = Depends(get_async_session)):
    """
    Retorna as cidades com menor número de casos confirmados acumulados.
    """
    try:
        result = await stats_service.get_least_affected_cities(limit, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {"data": result}

    except SQLAlchemyError as e:
        logger.exception(f"Erro de banco de dados ao buscar cidades menos afetadas: {e}")
        raise HTTPException(status_code=500, detail="Erro ao acessar o banco de dados.")

    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar cidades menos afetadas: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

# -----------------------------------------------------------
# Intervalos de confiança (JSON)
# -----------------------------------------------------------
@router.get("/confidence/cases")
async def confidence_interval_cases(
    confidence: float = Query(0.95, ge=0.8, le=0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Retorna o intervalo de confiança da média de novos casos diários.
    """
    try:
        return await stats_service.get_confidence_interval_cases(db, confidence)
    except Exception as e:
        logger.exception(f"Erro ao calcular IC de casos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao calcular intervalo de confiança: {str(e)}")


@router.get("/confidence/deaths")
async def confidence_interval_deaths(
    confidence: float = Query(0.95, ge=0.8, le=0.99),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Retorna o intervalo de confiança da média de novas mortes diárias.
    """
    try:
        return await stats_service.get_confidence_interval_deaths(db, confidence)
    except Exception as e:
        logger.exception(f"Erro ao calcular IC de mortes: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao calcular intervalo de confiança: {str(e)}")

# ==========================================================
# Endpoints de Dados para Gráficos (JSON)
# ==========================================================

@router.get("/data/time-series")
async def time_series_data(
    metric: str = Query("new_confirmed"),
    state: str | None = Query(None),
    db: AsyncSession = Depends(get_async_session)  # <-- CORREÇÃO: Padronizado para 'db'
):
    """
    Retorna dados de série temporal em JSON.
    """
    return await stats_service.get_time_series_data(db, metric, state)


# ==========================================================
# Endpoints de Imagem (PNG)
# ==========================================================

@router.get("/plot/histogram")
async def histogram_plot(
    metric: str = Query("new_confirmed"),
    bin_width: int = Query(100),
    max_value: int = Query(10000),
    db: AsyncSession = Depends(get_async_session)  # <-- CORREÇÃO: Padronizado para 'db'
):
    """
    Retorna um gráfico (plot) de histograma em formato PNG.
    (OBS: Funcionalmente idêntico a /image/histogram)
    """
    result = await stats_service.get_histogram_image(
        db, metric, bin_width, max_value
    )

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    if not isinstance(result, io.BytesIO) or result.getbuffer().nbytes == 0:
        logger.error(f"Erro ao gerar /plot/histogram (buffer vazio) para metric={metric}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar o gráfico.")

    return StreamingResponse(result, media_type="image/png")


@router.get("/image/time-series")
async def time_series_chart(
    metric: str = Query("new_confirmed"),
    state: str | None = Query(None),
    db: AsyncSession = Depends(get_async_session)  # <-- CORREÇÃO: Padronizado para 'db'
):
    """
    Retorna gráfico PNG de série temporal.
    """
    img = await stats_service.get_time_series_image(db, metric, state)

    # <-- CORREÇÃO: Tratamento de erro adequado
    if isinstance(img, dict) and "error" in img:
        raise HTTPException(status_code=404, detail=img["error"])
    
    # <-- CORREÇÃO: Verificação de buffer vazio
    if not isinstance(img, io.BytesIO) or img.getbuffer().nbytes == 0:
        logger.error(f"Erro ao gerar /image/time-series (buffer vazio) para metric={metric}, state={state}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar o gráfico.")

    return StreamingResponse(img, media_type="image/png")


@router.get("/image/histogram")
async def histogram_chart(
    metric: str = Query("new_confirmed"),
    bin_width: int = Query(100),
    max_value: int = Query(10000),
    db: AsyncSession = Depends(get_async_session)  # <-- CORREÇÃO: Padronizado para 'db'
):
    """
    Retorna gráfico PNG de histograma.
    (OBS: Funcionalmente idêntico a /plot/histogram)
    """
    img = await stats_service.get_histogram_image(db, metric, bin_width, max_value)

    # <-- CORREÇÃO: Tratamento de erro adequado
    if isinstance(img, dict) and "error" in img:
        raise HTTPException(status_code=404, detail=img["error"])
    
    # <-- CORREÇÃO: Verificação de buffer vazio
    if not isinstance(img, io.BytesIO) or img.getbuffer().nbytes == 0:
        logger.error(f"Erro ao gerar /image/histogram (buffer vazio) para metric={metric}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar o gráfico.")

    return StreamingResponse(img, media_type="image/png")