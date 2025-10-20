import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from typing import AsyncGenerator

from src.models.casos_covid import ModelBase

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("A variável de ambiente DATABASE_URL não foi definida ou lida corretamente!")

# -------------------------
# Async Engine & Session
# -------------------------
async_engine: AsyncEngine = create_async_engine(url = DB_URL, echo = False)

AsyncSessionLocal = sessionmaker(
    bind = async_engine,
    class_ = AsyncSession,
    expire_on_commit = False,
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency para FastAPI: retorna uma sessão assíncrona e garante o fechamento."""
    async with AsyncSessionLocal() as session:
        yield session

async def create_tables_async() -> None:
    """Cria tabelas usando o engine assíncrono."""
    print("Criando tabelas no modo async...")
    async with async_engine.begin() as conn:
        await conn.run_sync(ModelBase.metadata.create_all)
    print("Tabelas criadas com sucesso (async).")

def get_async_session_instance() -> AsyncSession:
    """Retorna uma instância de sessão assíncrona fora de FastAPI."""
    return AsyncSessionLocal()

# -------------------------
# Sync Engine & Session
# -------------------------

sync_db_url = DB_URL.replace("postgresql+asyncpg", "postgresql+psycopg2")
sync_engine: Engine = create_engine(url = sync_db_url, echo = False)

SyncSessionLocal = sessionmaker(
    bind = sync_engine,
    class_ = Session,
    expire_on_commit = False,
)

def get_sync_session() -> Session:
    """Retorna uma sessão síncrona para uso em ETL ou scripts."""
    return SyncSessionLocal()

def create_tables_sync() -> None:
    """Cria tabelas usando o engine síncrono."""
    print("Criando tabelas no modo sync...")
    ModelBase.metadata.create_all(bind = sync_engine)
    print("Tabelas criadas com sucesso (sync).")
