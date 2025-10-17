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

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações e limpezas necessárias aos dados.
    CORRIGIDO: Agora processa CIDADES em vez de apenas estados.
    """
    print("\n--- Etapa de Transformação (T) ---")
    print("Iniciando limpeza e transformação dos dados...")

    df_copy = df.copy()

    if 'city' in df_copy.columns:
        print("Normalizando nomes de cidades (minúsculas e sem acentos)...")
        df_copy['city'] = df_copy['city'].apply(
            lambda x: unidecode(x).lower() if pd.notna(x) else x
        )
        print("Normalização de cidades concluída.")

    df_copy['date'] = pd.to_datetime(df_copy['date'], errors = 'coerce')

    print("Filtrando dados para manter apenas registros a nível de CIDADE...")
    df_cities = df_copy[df_copy['place_type'] == 'city'].copy()
    
    print("Removendo registros com city vazio ou 'N/A'...")
    initial_count = len(df_cities)
    df_cities = df_cities[
        df_cities['city'].notna() & 
        (df_cities['city'] != '') & 
        (df_cities['city'] != 'n/a')
    ].copy()
    removed_count = initial_count - len(df_cities)
    print(f"Removidos {removed_count} registros com city inválido. Restantes: {len(df_cities)}")

    print("Imputando datas ausentes por interpolação...")
    df_cities = df_cities.sort_values(by = ['state', 'city', 'date'])
    df_cities['date'] = df_cities.groupby(['state', 'city'])['date'].transform(
        lambda x: x.interpolate(method = 'linear').ffill().bfill()
    )
    df_cities.dropna(subset = ['date'], inplace = True)
    df_cities['date'] = df_cities['date'].dt.date
    print("Imputação de datas concluída.")

    print("Imputando valores de população ausentes...")
    cols_to_fill_pop = ['estimated_population', 'estimated_population_2019']
    df_cities[cols_to_fill_pop] = df_cities.groupby(['state', 'city'])[cols_to_fill_pop].transform(
        lambda x: x.ffill().bfill()
    )
    df_cities.dropna(subset = cols_to_fill_pop, inplace = True)
    print("Imputação de população concluída.")

    target_col = 'last_available_confirmed_per_100k_inhabitants'
    if df_cities[target_col].isnull().any():
        print(f"Imputando a coluna '{target_col}' com KNNImputer...")
        df_cities['date_ordinal'] = df_cities['date'].apply(lambda x: x.toordinal())
        
        features_for_knn = [
            'date_ordinal', 'estimated_population', 'last_available_confirmed',
            'new_confirmed', 'last_available_deaths', target_col
        ]
        
        helper_cols = [col for col in features_for_knn if col != target_col]
        df_cities[helper_cols] = df_cities.groupby(['state', 'city'])[helper_cols].transform(
            lambda x: x.ffill().bfill()
        )
        df_cities.dropna(subset = helper_cols, inplace = True)

        imputed_dfs = []
        for (state, city), group in df_cities.groupby(['state', 'city']):
            group_copy = group.copy()
            original_index = group_copy.index
            group_features = group_copy[features_for_knn]
            
            if group_features.isnull().values.any():
                scaler = MinMaxScaler()
                group_scaled = pd.DataFrame(
                    scaler.fit_transform(group_features), 
                    columns = features_for_knn
                )
                
                imputer = KNNImputer(n_neighbors=5)
                group_imputed_scaled = pd.DataFrame(
                    imputer.fit_transform(group_scaled), 
                    columns = features_for_knn
                )
                
                group_imputed = pd.DataFrame(
                    scaler.inverse_transform(group_imputed_scaled),
                    columns = features_for_knn,
                    index = original_index
                )
                imputed_dfs.append(group_imputed)
            else:
                imputed_dfs.append(group_features.set_index(original_index))

        if imputed_dfs:
            df_imputed_full = pd.concat(imputed_dfs)
            df_cities[target_col] = df_imputed_full[target_col]

        df_cities.drop(columns=['date_ordinal'], inplace=True)
        print("Imputação com KNN concluída.")

    df_cities['city_ibge_code'] = pd.to_numeric(
        df_cities['city_ibge_code'], 
        errors = 'coerce'
    ).astype('Int64')
    
    df_cities.rename(columns = {'date': 'datetime'}, inplace = True)

    model_columns = [c.name for c in CasoCovid.__table__.columns if c.name != 'id']
    df_to_load = df_cities.reindex(columns = model_columns)
    
    if 'city' in df_to_load.columns:
        print(f"Coluna 'city' presente com {df_to_load['city'].notna().sum()} valores válidos")
    else:
        print("AVISO: Coluna 'city' não está no DataFrame final!")
    
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