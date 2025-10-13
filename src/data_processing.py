import pandas as pd
from database import sync_engine
from src.models.casos_covid import ModelBase, CasoCovid
from sqlalchemy import text, inspect
from pathlib import Path
import requests
import io 

DATA_PATH = "data/caso_full.csv"
TABLE_NAME = CasoCovid.__tablename__

def etl_pipeline():
    print("Iniciando o processo de ETL...")

    print("Verificando se a base de dados já está populada...")
    inspector = inspect(sync_engine)
    if inspector.has_table(TABLE_NAME):
        with sync_engine.connect() as connection:
            query = text(f"SELECT COUNT(1) FROM {TABLE_NAME};")
            count = connection.execute(query).scalar()
            if count and count > 0:
                print(f"A tabela '{TABLE_NAME}' já contém {count} registos. Pulando o processo de ETL.")
                return
    else:
        print(f"A tabela '{TABLE_NAME}' ainda não existe. O ETL continuará.")

    print("Verificando e criando a tabela no banco de dados, se necessário...")
    ModelBase.metadata.create_all(bind = sync_engine)
    print("Estrutura da tabela garantida.")

    if Path(DATA_PATH).exists():
        print(f"Lendo dados do arquivo local: {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            compressed_dataset_url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
            print(f"Arquivo local não encontrado. Baixando de {compressed_dataset_url}...")
            
            response = requests.get(compressed_dataset_url)
            response.raise_for_status()

            print("Download concluído. Descompactando e lendo os dados para o DataFrame...")
            df = pd.read_csv(io.BytesIO(response.content), compression = 'gzip')

            print(f"Salvando uma cópia local em {DATA_PATH} para acelerar futuras execuções...")
            Path(DATA_PATH).parent.mkdir(parents = True, exist_ok = True)
            df.to_csv(DATA_PATH, index = False)
            print("Cópia local salva com sucesso.")

        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar ou processar o arquivo: {e}")
            return 
    print("Iniciando limpeza e transformação dos dados...")
    df_states = df[df['place_type'] == 'state'].copy()
    df_states['date'] = pd.to_datetime(df_states['date']).dt.date

    df_states.rename(columns={'date': 'datetime'}, inplace = True)
    
    df_states['city_ibge_code'] = pd.to_numeric(df_states['city_ibge_code'], errors = 'coerce').astype('Int64')

    model_columns = [c.name for c in CasoCovid.__table__.columns if c.name != 'id']
    df_to_load = df_states.reindex(columns = model_columns)

    with sync_engine.connect() as connection:
        with connection.begin():
            print(f"Limpando a tabela '{TABLE_NAME}' antes da inserção...")
            connection.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))
        print(f"Tabela '{TABLE_NAME}' foi limpa.")

    print(f"Inserindo {len(df_to_load)} registos na tabela '{TABLE_NAME}'...")
    df_to_load.to_sql(TABLE_NAME, con = sync_engine, if_exists = 'append', index = False, chunksize = 1000)
    print("Dados inseridos com sucesso!")

if __name__ == "__main__":
    etl_pipeline()