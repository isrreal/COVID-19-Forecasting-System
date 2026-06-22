import logging
from datetime import date, datetime

import pandas as pd
import pysus
from pandas import DataFrame
from sqlalchemy import inspect, text

from database import create_tables, sync_engine
from src.models.caso_dengue import CasoDengue

TABLE_NAME = CasoDengue.__tablename__
SINAN_DISEASE = "DENG"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_BLANK_CODES = {"", "0", "00"}


def _parse_date(value: str) -> date | None:
    s = str(value).strip()
    if not s or len(s) != 8:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_int_code(value: str | int) -> int | None:
    s = str(value).strip()
    if s in _BLANK_CODES:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# -----------------------------
# Extraction
# -----------------------------
def extract_data(years: list[int]) -> DataFrame:
    logger.info("--- Extraction Stage ---")
    parquet_files: list[str] = pysus.sinan(SINAN_DISEASE, years)

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    logger.info(f"Extracted {len(df)} raw records from {years}.")
    return df


# -----------------------------
# Transformation
# -----------------------------
def transform(df: DataFrame) -> DataFrame:
    logger.info("--- Transformation Stage ---")

    raw_cols = [
        "DT_NOTIFIC",
        "SEM_NOT",
        "NU_ANO",
        "SG_UF_NOT",
        "ID_MUNICIP",
        "CLASSI_FIN",
        "EVOLUCAO",
        "HOSPITALIZ",
        "SOROTIPO",
        "DT_OBITO",
        "CS_SEXO",
        "NU_IDADE_N",
    ]
    df = df[raw_cols].copy()

    df["dt_notific"] = df["DT_NOTIFIC"].astype(str).apply(_parse_date)
    df = df[df["dt_notific"].notna()].copy()

    df["sem_not"] = df["SEM_NOT"].astype(str).str.strip()
    df["nu_ano"] = pd.to_numeric(df["NU_ANO"], errors="coerce").astype("Int16")
    df["sg_uf_not"] = pd.to_numeric(df["SG_UF_NOT"], errors="coerce").astype("Int16")
    df["id_municip"] = pd.to_numeric(df["ID_MUNICIP"], errors="coerce").astype("Int32")

    for raw, out in [
        ("CLASSI_FIN", "classi_fin"),
        ("EVOLUCAO", "evolucao"),
        ("HOSPITALIZ", "hospitaliz"),
        ("SOROTIPO", "sorotipo"),
    ]:
        df[out] = df[raw].astype(str).apply(_parse_int_code)

    df["dt_obito"] = df["DT_OBITO"].astype(str).apply(_parse_date)

    # Keep only M/F; ignore and empty become None
    df["cs_sexo"] = df["CS_SEXO"].astype(str).str.strip()
    df["cs_sexo"] = df["cs_sexo"].where(df["cs_sexo"].isin(["M", "F"]), other=None)

    df["nu_idade_n"] = pd.to_numeric(df["NU_IDADE_N"], errors="coerce").astype("Int32")

    output_cols = [
        "dt_notific",
        "sem_not",
        "nu_ano",
        "sg_uf_not",
        "id_municip",
        "classi_fin",
        "evolucao",
        "hospitaliz",
        "sorotipo",
        "dt_obito",
        "cs_sexo",
        "nu_idade_n",
    ]
    df = df[output_cols].where(pd.notna(df[output_cols]), other=None)

    logger.info(f"Transformed {len(df)} records.")
    return df


# -----------------------------
# Loading
# -----------------------------
def load_data(df: DataFrame) -> None:
    logger.info("--- Loading Stage ---")
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to insert.")
        return

    with sync_engine.begin() as conn:
        logger.info(f"Truncating '{TABLE_NAME}'...")
        conn.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))

        logger.info(f"Inserting {len(df)} rows...")
        df.to_sql(
            TABLE_NAME,
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10_000,
        )
        logger.info("Data loaded successfully.")


# -----------------------------
# ETL Pipeline
# -----------------------------
def main_etl_pipeline(years: list[int] | None = None) -> None:
    if years is None:
        years = [2023, 2024]

    logger.info("Starting ETL pipeline...")
    inspector = inspect(sync_engine)

    if inspector.has_table(TABLE_NAME):
        with sync_engine.connect() as conn:
            row_count = conn.execute(
                text(f'SELECT COUNT(1) FROM "{TABLE_NAME}";')
            ).scalar()
            if row_count and row_count > 0:
                logger.info(
                    f"Table '{TABLE_NAME}' already has {row_count} rows. Skipping ETL."
                )
                return
    else:
        create_tables()
        logger.info("Table created.")

    df_raw = extract_data(years)
    df_clean = transform(df_raw)
    load_data(df_clean)

    logger.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main_etl_pipeline()
