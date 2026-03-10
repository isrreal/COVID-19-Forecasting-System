import logging
from pathlib import Path
import io
import requests

import pandas
from pandas import DataFrame
import numpy

from unidecode import unidecode
from sqlalchemy import text, inspect

from database import sync_engine, create_tables
from src.models.casos_covid import CasoCovid

DATA_PATH = "data/caso_full.csv"
TABLE_NAME = CasoCovid.__tablename__
DATASET_URL = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Extraction
# -----------------------------
def extract_data(data_path: str, url: str) -> DataFrame | None:
    logging.info("--- Extraction Stage ---")
    path = Path(data_path)

    if path.exists():
        logging.info(f"Reading local file: {data_path}")
        return pandas.read_csv(path)

    logging.info("Downloading dataset...")
    response = requests.get(url)
    response.raise_for_status()
    dataframe = pandas.read_csv(io.BytesIO(response.content), compression = "gzip")

    path.parent.mkdir(parents = True, exist_ok = True)
    dataframe.to_csv(path, index = False)

    return dataframe


# -----------------------------
# Cleaning & Conversion
# -----------------------------
def clean_and_convert(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()

    # Normalize city names
    if "city" in dataframe.columns:
        dataframe["city"] = dataframe["city"].astype("category")

    # Convert numeric columns
    for col in ["estimated_population", "estimated_population_2019", "city_ibge_code"]:
        if col in dataframe.columns:
            dataframe[col] = pandas.to_numeric(dataframe[col], errors = "coerce").astype("Int32")

    # Convert place_type
    if "place_type" in dataframe.columns:
        city_normalized = unidecode("City")
        state_normalized = unidecode("State")

        dataframe["place_type"] = dataframe["place_type"].replace(
            {city_normalized: "C", state_normalized: "S"}
        ).astype("category")

    # Replace broken null values
    dataframe.replace(
        ["<NA>", "NA", "NaN", "nan", "null", "", "Importados/Indefinidos"],
        numpy.nan,
        inplace = True
    )

    # Filter invalid cities
    if "city" in dataframe.columns:
        initial_count = len(dataframe)
        dataframe = dataframe[dataframe["city"].notna()].copy()
        removed = initial_count - len(dataframe)

        logging.info(
            f"Removed {removed} rows with invalid city values. Remaining: {len(dataframe)}"
        )

    # Convert dates
    if "date" in dataframe.columns and "last_available_date" in dataframe.columns:
        dataframe["date"] = pandas.to_datetime(dataframe["date"], errors = "coerce")
        dataframe["last_available_date"] = pandas.to_datetime(
            dataframe["last_available_date"], errors = "coerce"
        )

        dataframe.sort_values(["state", "city", "date"], inplace = True)

    # Recalculate cases per 100k
    if (
        "last_available_confirmed" in dataframe.columns
        and "estimated_population" in dataframe.columns
    ):
        dataframe["confirmed_per_100k"] = (
            dataframe["last_available_confirmed"]
            / dataframe["estimated_population"]
        ) * 100000

    # Drop old column
    if "last_available_confirmed_per_100k_inhabitants" in dataframe.columns:
        dataframe.drop(
            columns = ["last_available_confirmed_per_100k_inhabitants"],
            inplace = True
        )

    return dataframe


# -----------------------------
# Analysis
# -----------------------------
def analyze_missing_columns(dataframe: DataFrame, stage: str) -> None:
    logging.info(f"--- Missing Values Analysis (Stage: {stage}) ---")
    missing_counts = dataframe.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        logging.info("No missing values found.")
        return

    total_rows = len(dataframe)
    summary = DataFrame({
        "missing_count": missing_counts,
        "missing_rate": (missing_counts / total_rows) * 100
    }).sort_values("missing_rate", ascending = False)

    summary["missing_rate"] = summary["missing_rate"].map("{:.2f}%".format)

    logging.info(f"{len(summary)} columns with missing values:\n{summary}")


# -----------------------------
# Prepare for ORM
# -----------------------------
def prepare_for_orm(dataframe: DataFrame, model_cls) -> DataFrame:
    model_columns = [c.name for c in model_cls.__table__.columns if c.name != "id"]
    dataframe_final = dataframe.copy()
    dataframe_final.rename(columns = {"date": "datetime"}, inplace = True)
    dataframe_final = dataframe_final.reindex(columns = model_columns)
    return dataframe_final

# -----------------------------
# Loading
# -----------------------------
def load_data(dataframe: DataFrame) -> None:
    logging.info("--- Loading Stage ---")
    if dataframe.empty:
        logging.warning("DataFrame is empty. Nothing to insert.")
        return

    try:
        with sync_engine.begin() as conn:
            logging.info(f"Truncating table '{TABLE_NAME}'...")
            conn.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))
            logging.info(f"Table '{TABLE_NAME}' cleared.")

            logging.info(f"Inserting {len(dataframe)} rows into '{TABLE_NAME}'...")
            dataframe.to_sql(
                TABLE_NAME,
                con = conn,
                if_exists = "append",
                index = False,
                method = "multi",
                chunksize = 10_000
            )
            logging.info("Data loaded successfully!")

    except Exception as e:
        logging.error(f"Error inserting data: {e}")


# -----------------------------
# ETL Pipeline
# -----------------------------
def main_etl_pipeline():
    logging.info("Starting ETL pipeline...")

    inspector = inspect(sync_engine)

    # Skip ETL if table exists with data
    if inspector.has_table(TABLE_NAME):
        with sync_engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(1) FROM {TABLE_NAME};")).scalar()
            if row_count and row_count > 0:
                logging.info(f"Table '{TABLE_NAME}' already has {row_count} rows. Skipping ETL.")
                return
    else:
        create_tables()
        logging.info("Table structure ensured.")

    # Extraction
    dataframe = extract_data(DATA_PATH, DATASET_URL)
    if dataframe is None or dataframe.empty:
        logging.error("Data extraction failed. Aborting ETL.")
        return
    logging.info(f"Extracted dataset with {len(dataframe)} rows.")

    analyze_missing_columns(dataframe, stage = "Raw")

    # Cleaning / Transformation
    dataframe_transformed = clean_and_convert(dataframe)

    analyze_missing_columns(dataframe_transformed, stage = "Transformed")
    
    logging.info(f"Transformed dataset with {len(dataframe_transformed)} rows.")

    # Load
    # load_data(dataframe_transformed)

    logging.info("ETL pipeline completed successfully!")

if __name__ == "__main__":
    main_etl_pipeline()