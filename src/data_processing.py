import pandas as pd
import numpy as np
from database import sync_engine
from src.models.casos_covid import ModelBase, CasoCovid
from sqlalchemy import text, inspect
from pathlib import Path
import requests
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from unidecode import unidecode

DATA_PATH = "data/caso_full.csv"
TABLE_NAME = CasoCovid.__tablename__
DATASET_URL = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"

def colunas_vazias(df: pd.DataFrame, stage: str):
    """
    Analisa um DataFrame em um determinado estágio (ex: 'bruto', 'transformado')
    e imprime um relatório de valores faltantes.
    """
    print(f"\n--- Análise de Colunas Vazias (Estágio: {stage}) ---")
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("Ótima notícia! Nenhuma coluna com valores vazios foi encontrada.")
        print("--------------------------------------------------")
        return

    total_rows = len(df)
    missing_df = pd.DataFrame({
        'Valores Faltantes': missing_counts,
        '% Faltante': (missing_counts / total_rows) * 100
    })

    missing_df = missing_df.sort_values(by = '% Faltante', ascending=False)
    missing_df['% Faltante'] = missing_df['% Faltante'].map('{:.2f}%'.format)

    print(f"Encontradas {len(missing_df)} colunas com valores faltantes (de um total de {total_rows} linhas):")
    print(missing_df)
    print("--------------------------------------------------")

def extract_data(data_path: str, url: str) -> pd.DataFrame | None:
    """
    Extrai os dados, seja de um arquivo local ou fazendo o download.
    Retorna um DataFrame ou None em caso de erro.
    """
    print("\n--- Etapa de Extração (E) ---")
    if Path(data_path).exists():
        print(f"Lendo dados do arquivo local: {data_path}...")
        return pd.read_csv(data_path)
    
    try:
        print(f"Arquivo local não encontrado. Baixando de {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        print("Download concluído. Descompactando e lendo os dados...")
        df = pd.read_csv(io.BytesIO(response.content), compression = 'gzip')
        
        print(f"Salvando uma cópia local em {data_path}...")
        Path(data_path).parent.mkdir(parents = True, exist_ok = True)
        df.to_csv(data_path, index = False)
        print("Cópia local salva com sucesso.")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar ou processar o arquivo: {e}")
        return None
def normalize_city_names(df: pd.DataFrame, city_col: str = 'city') -> pd.DataFrame:
    if city_col in df.columns:
        df[city_col] = df[city_col].astype(str).str.strip().str.lower().map(unidecode)
    return df

def filter_valid_cities(df: pd.DataFrame, city_col: str = 'city') -> pd.DataFrame:
    initial_count = len(df)
    df[city_col] = df[city_col].replace(['', 'n/a', 'na', 'nan'], pd.NA)
    df_filtered = df[df[city_col].notna()].copy()
    removed_count = initial_count - len(df_filtered)
    print(f"Removidos {removed_count} registros com city inválido. Restantes: {len(df_filtered)}")
    return df_filtered

def impute_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['state', 'city', 'date'])
    df['date'] = df.groupby(['state', 'city'])['date'].transform(
        lambda x: x.interpolate(method = 'linear').ffill().bfill()
    )
    df.dropna(subset = ['date'], inplace = True)
    df['date'] = df['date'].dt.date
    return df

def impute_population(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_fill_pop = ['estimated_population', 'estimated_population_2019']
    df[cols_to_fill_pop] = df.groupby(['state', 'city'])[cols_to_fill_pop].transform(
        lambda x: x.ffill().bfill()
    )
    df.dropna(subseb = cols_to_fill_pop, inplace = True)
    return df

def knn_impute(df: pd.DataFrame, target_col: str, features_for_knn: list, group_cols = ['state','city'], n_neighbors = 5) -> pd.DataFrame:
    df['date_ordinal'] = pd.to_datetime(df['date']).view('int64') // 10 ** 19
    helper_cols = [col for col in features_for_knn if col != target_col]
    
    df[helper_cols] = df.groupby(group_cols)[helper_cols].ffill().bfill()
    df.dropna(subset = helper_cols, inplace = True)
    
    imputed_dfs = []
    for (_, _), group in df.groupby(group_cols):
        group = group.copy()
        index = group.index
        features = group[features_for_knn]
        if features.isnull().values.any():
            scaler = MinMaxScaler()
            scaled = pd.DataFrame(scaler.fit_transform(features), columns = features_for_knn, index = index)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_scaled = pd.DataFrame(imputer.fit_transform(scaled), columns=features_for_knn, index = index)
            group_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns = features_for_knn, index = index)
            imputed_dfs.append(group_imputed)
        else:
            imputed_dfs.append(features)
    df[target_col] = pd.concat(imputed_dfs)[target_col]
    df.drop(columns = ['date_ordinal'], inplace = True)
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
    print("\n--- Etapa de Transformação (T) ---")
    
    df_copy = df.copy()
    df_copy = normalize_city_names(df_copy)
    
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
    
    print(f"Transformação de dados concluída. Total de registros: {len(df_to_load)}")
    return df_to_load

def load_data(df: pd.DataFrame, table_name: str, engine):
    """
    Carrega o DataFrame transformado no banco de dados.
    """
    print("\n--- Etapa de Carga (L) ---")
    with engine.connect() as connection:
        with connection.begin():
            print(f"Limpando a tabela '{table_name}' antes da inserção...")
            connection.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY;'))
        print(f"Tabela '{table_name}' foi limpa.")

    print(f"Inserindo {len(df)} registros na tabela '{table_name}'...")
    df.to_sql(
        table_name,
        con = engine,
        if_exists = 'append',
        index = False,
        chunksize = 1000
    )
    print("Dados inseridos com sucesso!")

def main_etl_pipeline():
    """
    Função principal que orquestra todo o processo de ETL.
    """
    print("Iniciando o processo de ETL...")

    inspector = inspect(sync_engine)
    if inspector.has_table(TABLE_NAME):
        with sync_engine.connect() as connection:
            query = text(f"SELECT COUNT(1) FROM {TABLE_NAME};")
            count = connection.execute(query).scalar()
            if count and count > 0:
                print(f"A tabela '{TABLE_NAME}' já contém {count} registros. Pulando o processo de ETL.")
                return

    print(f"A tabela '{TABLE_NAME}' ainda não existe ou está vazia. O ETL continuará.")
    ModelBase.metadata.create_all(bind = sync_engine)
    print("Estrutura da tabela no banco de dados garantida.")

    df_raw = extract_data(data_path = DATA_PATH, url = DATASET_URL)
    if df_raw is None:
        print("Falha na extração de dados. Abortando o pipeline.")
        return
    
    print(f"\nEstatísticas do dataset bruto:")
    print(f"  Total de registros: {len(df_raw)}")
    if 'place_type' in df_raw.columns:
        print(f"  Registros de cidades: {(df_raw['place_type'] == 'city').sum()}")
        print(f"  Registros de estados: {(df_raw['place_type'] == 'state').sum()}")
    
    colunas_vazias(df_raw, stage = "Dados Brutos")

    df_transformed = transform_data(df_raw)
    colunas_vazias(df_transformed, stage = "Dados Transformados")

    load_data(df_transformed, table_name = TABLE_NAME, engine = sync_engine)

    print("\nProcesso de ETL concluído com sucesso.")

if __name__ == "__main__":
    main_etl_pipeline()