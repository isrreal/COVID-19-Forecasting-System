import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

from src import data_processing
from src import train 

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

def main(args):
    """
    Função principal que orquestra o pipeline de ETL e treinamento.
    """
    start_time = time.time()
    logging.info("Iniciando o workflow de Machine Learning.")

    if not args.skip_etl:
        logging.info("Iniciando a Etapa 1: Pipeline de ETL de dados.")
        try:
            data_processing.main_etl_pipeline()
            logging.info("Etapa 1 concluída com sucesso.")
        except Exception as e:
            logging.error(f"Falha na Etapa 1 (ETL): {e}")
            return 

    states_to_train = args.states
    if not states_to_train:
        logging.warning("Nenhum estado especificado para treinamento. Finalizando workflow.")
        return

    logging.info(f"Iniciando a Etapa 2: Treinamento para os estados: {', '.join(states_to_train)}")

    if args.parallel and len(states_to_train) > 1:
        logging.info(f"Executando treinamento em paralelo para {len(states_to_train)} estados.")
        with ProcessPoolExecutor() as executor:
            executor.map(train.run_experiments, states_to_train)
    else:
        logging.info("Executando treinamento em modo sequencial.")
        for state_code in states_to_train:
            train.run_experiments(state = state_code)
    
    logging.info("Etapa 2 concluída com sucesso.")
    end_time = time.time()
    logging.info(f"Workflow concluído em {end_time - start_time:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Orquestrador do Pipeline de ML para Forecasting de COVID-19.")
    parser.add_argument(
        '--states', nargs = '+', required = True,
        help = "Lista de siglas dos estados para treinar (ex: CE SP RJ)."
    )
    parser.add_argument(
        '--skip-etl', action = 'store_true',
        help = "Pula a etapa de ETL se os dados já estiverem atualizados."
    )
    parser.add_argument(
        '--parallel', action = 'store_true',
        help = "Executa o treinamento para múltiplos estados em paralelo."
    )
    
    args = parser.parse_args()
    main(args)
