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


def _parse_date(raw_value: str) -> date | None:
    value_str = str(raw_value).strip()
    if not value_str or len(value_str) != 8:
        return None
    try:
        return datetime.strptime(value_str, "%Y%m%d").date()
    except ValueError:
        return None


def _parse_int_code(raw_value: str | int) -> int | None:
    value_str = str(raw_value).strip()
    if value_str in _BLANK_CODES:
        return None
    try:
        return int(value_str)
    except ValueError:
        return None


# -----------------------------
# Extraction
# -----------------------------
def extract_data(years: list[int]) -> DataFrame:
    logger.info("--- Extraction Stage ---")
    parquet_paths: list[str] = pysus.sinan(SINAN_DISEASE, years)

    yearly_frames = [pd.read_parquet(path) for path in parquet_paths]
    raw_df = pd.concat(yearly_frames, ignore_index=True)

    logger.info(f"Extracted {len(raw_df)} raw records from {years}.")
    return raw_df


# -----------------------------
# Transformation
# -----------------------------
def transform(raw_df: DataFrame) -> DataFrame:
    logger.info("--- Transformation Stage ---")

    sinan_cols = [
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
    dengue_df = raw_df[sinan_cols].copy()

    dengue_df["dt_notific"] = dengue_df["DT_NOTIFIC"].astype(str).apply(_parse_date)
    dengue_df = dengue_df[dengue_df["dt_notific"].notna()].copy()

    dengue_df["sem_not"] = dengue_df["SEM_NOT"].astype(str).str.strip()
    dengue_df["nu_ano"] = pd.to_numeric(dengue_df["NU_ANO"], errors="coerce").astype(
        "Int16"
    )
    dengue_df["sg_uf_not"] = pd.to_numeric(
        dengue_df["SG_UF_NOT"], errors="coerce"
    ).astype("Int16")
    dengue_df["id_municip"] = pd.to_numeric(
        dengue_df["ID_MUNICIP"], errors="coerce"
    ).astype("Int32")

    categorical_cols = [
        ("CLASSI_FIN", "classi_fin"),
        ("EVOLUCAO", "evolucao"),
        ("HOSPITALIZ", "hospitaliz"),
        ("SOROTIPO", "sorotipo"),
    ]
    for sinan_col, model_col in categorical_cols:
        dengue_df[model_col] = dengue_df[sinan_col].astype(str).apply(_parse_int_code)

    dengue_df["dt_obito"] = dengue_df["DT_OBITO"].astype(str).apply(_parse_date)

    # Keep only M/F; ignore and empty become None
    dengue_df["cs_sexo"] = dengue_df["CS_SEXO"].astype(str).str.strip()
    dengue_df["cs_sexo"] = dengue_df["cs_sexo"].where(
        dengue_df["cs_sexo"].isin(["M", "F"]), other=None
    )

    dengue_df["nu_idade_n"] = pd.to_numeric(
        dengue_df["NU_IDADE_N"], errors="coerce"
    ).astype("Int32")

    model_cols = [
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
    clean_df = dengue_df[model_cols].where(pd.notna(dengue_df[model_cols]), other=None)

    logger.info(f"Transformed {len(clean_df)} records.")
    return clean_df


# -----------------------------
# Loading
# -----------------------------
def load_data(clean_df: DataFrame) -> None:
    logger.info("--- Loading Stage ---")
    if clean_df.empty:
        logger.warning("DataFrame is empty. Nothing to insert.")
        return

    with sync_engine.begin() as conn:
        logger.info(f"Truncating '{TABLE_NAME}'...")
        conn.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))

        logger.info(f"Inserting {len(clean_df)} rows...")
        clean_df.to_sql(
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
            existing_row_count = conn.execute(
                text(f'SELECT COUNT(1) FROM "{TABLE_NAME}";')
            ).scalar()
            if existing_row_count and existing_row_count > 0:
                logger.info(
                    f"Table '{TABLE_NAME}' already has {existing_row_count} rows. Skipping ETL."
                )
                return
    else:
        create_tables()
        logger.info("Table created.")

    raw_df = extract_data(years)
    clean_df = transform(raw_df)
    load_data(clean_df)

    logger.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main_etl_pipeline()
