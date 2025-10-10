import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.models.casos_covid import ModelBase

# load_dotenv() já não é estritamente necessário se o docker-compose injetar a variável
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise ValueError("A variável de ambiente DATABASE_URL não foi definida ou lida corretamente!")

async_engine: AsyncEngine = create_async_engine(url=DB_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind = async_engine,
    class_ = AsyncSession,
    expire_on_commit = False,
)

sync_db_url = DB_URL.replace("postgresql+asyncpg", "postgresql+psycopg2")
sync_engine: Engine = create_engine(url=sync_db_url, echo=False)

def get_async_session() -> AsyncSession:
    """Retorna uma nova instância de sessão assíncrona da fábrica."""
    return AsyncSessionLocal()

async def create_tables_async() -> None:
    """Cria as tabelas no modo assíncrono."""
    print("Criando tabelas no banco de dados (modo async)...")
    async with async_engine.begin() as conn:
        await conn.run_sync(ModelBase.metadata.create_all)
    print("Tabelas criadas com sucesso.")

