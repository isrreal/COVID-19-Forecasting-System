import logging
from pathlib import Path
import io
import requests
from typing import List

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
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

def colunas_vazias(df: pd.DataFrame, stage: str) -> None:
    logging.info(f"--- Análise de Colunas Vazias (Estágio: {stage}) ---")
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        logging.info("Nenhuma coluna com valores vazios encontrada.")
        return

    total_rows = len(df)
    missing_df = pd.DataFrame({
        'Valores Faltantes': missing_counts,
        '% Faltante': (missing_counts / total_rows) * 100
    }).sort_values('% Faltante', ascending = False)
    missing_df['% Faltante'] = missing_df['% Faltante'].map('{:.2f}%'.format)

    logging.info(f"{len(missing_df)} colunas com valores faltantes (de {total_rows} linhas):\n{missing_df}")


def extract_data(data_path: str, url: str) -> pd.DataFrame | None:
    logging.info("--- Etapa de Extração (E) ---")
    if Path(data_path).exists():
        logging.info(f"Lendo dados do arquivo local: {data_path}")
        return pd.read_csv(data_path)

    try:
        logging.info(f"Arquivo local não encontrado. Baixando de {url}...")
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), compression = 'gzip')
        Path(data_path).parent.mkdir(parents = True, exist_ok = True)
        df.to_csv(data_path, index = False)
        logging.info(f"Cópia local salva em {data_path}")
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao baixar/processar o arquivo: {e}")
        return None


def normalize_city_names(df: pd.DataFrame, city_col: str = 'city') -> pd.DataFrame:
    df = df.copy()
    if city_col in df.columns:
        df[city_col] = df[city_col].astype(str).str.strip().str.lower().map(unidecode)
    return df


def filter_valid_cities(df: pd.DataFrame, city_col: str = 'city') -> pd.DataFrame:
    df = df.copy()
    initial_count = len(df)
    df[city_col] = df[city_col].replace(['', 'n/a', 'na', 'nan'], pd.NA)
    df_filtered = df[df[city_col].notna()].copy()
    removed_count = initial_count - len(df_filtered)
    logging.info(f"Removidos {removed_count} registros com city inválido. Restantes: {len(df_filtered)}")
    return df_filtered


def impute_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['state', 'city', 'date'])
    df['date'] = df.groupby(['state', 'city'])['date'].transform(
        lambda x: x.interpolate(method='linear').ffill().bfill()
    )
    df.dropna(subset=['date'], inplace=True)
    df['date'] = df['date'].dt.date
    return df


def impute_population(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_fill_pop = ['estimated_population', 'estimated_population_2019']
    df[cols_to_fill_pop] = df.groupby(['state', 'city'])[cols_to_fill_pop].transform(
        lambda x: x.ffill().bfill()
    )
    df.dropna(subset=cols_to_fill_pop, inplace=True)
    return df


def knn_impute(
        df: pd.DataFrame,
        target_col: str,
        features_for_knn: List[str],
        group_cols: List[str] = ['state', 'city'],
        n_neighbors: int = 5) -> pd.DataFrame:

    df = df.copy()
    df['date_ordinal'] = pd.to_datetime(df['date']).view('int64') // 10 ** 9
    helper_cols = [col for col in features_for_knn if col != target_col]
    df[helper_cols] = df.groupby(group_cols)[helper_cols].ffill().bfill()
    df.dropna(subset=helper_cols, inplace=True)

    imputed_values = []
    for _, group in df.groupby(group_cols):
        group = group.copy()
        index = group.index
        features = group[features_for_knn]
        if features.isnull().values.any():
            scaler = MinMaxScaler()
            scaled = pd.DataFrame(scaler.fit_transform(features), columns=features_for_knn, index=index)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_scaled = pd.DataFrame(imputer.fit_transform(scaled), columns=features_for_knn, index=index)
            group_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled),
                                         columns=features_for_knn, index=index)
            imputed_values.append(group_imputed)
        else:
            imputed_values.append(features)

    df[target_col] = pd.concat(imputed_values)[target_col]
    df.drop(columns=['date_ordinal'], inplace=True)
    return df


def prepare_for_orm(df: pd.DataFrame, model_cls) -> pd.DataFrame:
    model_columns = [c.name for c in model_cls.__table__.columns if c.name != 'id']
    df_final = (
        df
        .assign(city_ibge_code = lambda x: pd.to_numeric(x['city_ibge_code'], errors = 'coerce').astype('Int64'))
        .rename(columns = {'date': 'datetime'})
        .reindex(columns = model_columns)
    )
    return df_final


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("--- Etapa de Transformação (T) ---")    

    df_copy = normalize_city_names(df)
    print("DEBUG")
    df_cities = df_copy[df_copy['place_type'] == 'city'].copy()
    df_cities = filter_valid_cities(df_cities)
    df_cities['date'] = pd.to_datetime(df_cities['date'], errors = 'coerce')
    df_cities = impute_dates(df_cities)
    df_cities = impute_population(df_cities)
    target_col = 'last_available_confirmed_per_100k_inhabitants'
    if df_cities[target_col].isnull().any():
        features_for_knn = [
            'date_ordinal', 'estimated_population', 'last_available_confirmed',
            'new_confirmed', 'last_available_deaths', target_col
        ]
        df_cities = knn_impute(df_cities, target_col, features_for_knn)

    df_to_load = prepare_for_orm(df_cities, CasoCovid)
    logging.info(f"Transformação concluída. Total de registros: {len(df_to_load)}")
    return df_to_load


def load_data(df: pd.DataFrame) -> None:
    """
    Carrega o DataFrame transformado no banco de dados usando pandas.to_sql().
    """
    logging.info("--- Etapa de Carga (L) ---")

    if df.empty:
        logging.warning("DataFrame vazio. Nenhum dado será inserido.")
        return

    try:
        with sync_engine.begin() as connection:
            logging.info(f"Limpando tabela '{TABLE_NAME}'...")
            connection.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))
            logging.info(f"Tabela '{TABLE_NAME}' limpa com sucesso.")

            logging.info(f"Inserindo {len(df)} registros na tabela '{TABLE_NAME}' via pandas.to_sql()...")
            df.to_sql(
                TABLE_NAME,
                con = connection,
                if_exists = 'append',
                index = False,
                method = 'multi',        
                chunksize = 10_000       
            )

            logging.info("Dados inseridos com sucesso!")

    except Exception as e:
        logging.error(f"Erro ao carregar dados no banco: {e}")

def main_etl_pipeline():
    logging.info("Iniciando o pipeline ETL...")

    inspector = inspect(sync_engine)
    if inspector.has_table(TABLE_NAME):
        with sync_engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(1) FROM {TABLE_NAME};")).scalar()
            if count and count > 0:
                logging.info(f"Tabela '{TABLE_NAME}' já contém {count} registros. Pulando ETL.")
                return

    create_tables()

    logging.info("Estrutura da tabela garantida.")

    df_raw = extract_data(DATA_PATH, DATASET_URL)
    if df_raw is None:
        logging.error("Falha na extração. Abortando ETL.")
        return

    colunas_vazias(df_raw, stage = "Bruto")

    df_transformed = transform_data(df_raw)

    colunas_vazias(df_transformed, stage = "Transformado")

    load_data(df_transformed)

    logging.info("ETL concluído com sucesso!")

if __name__ == "__main__":
    main_etl_pipeline()
