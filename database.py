import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from src.models.caso_dengue import ModelBase

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine: Engine = create_engine(url=DB_URL, echo=False)

_SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

# keep the old name so existing imports don't break
sync_engine = engine


def get_sync_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a sync session and ensures it is closed."""
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    """Creates all tables using the sync engine."""
    ModelBase.metadata.create_all(bind=engine)
