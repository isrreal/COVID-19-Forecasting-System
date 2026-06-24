import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

from src import data_processing
from src import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    """Orchestrates the ETL and training pipeline."""
    start_time = time.time()
    logging.info("Starting dengue forecasting ML workflow.")

    if not args.skip_etl:
        logging.info("Step 1: Running ETL pipeline.")
        try:
            data_processing.main_etl_pipeline()
            logging.info("Step 1 completed successfully.")
        except Exception as e:
            logging.error(f"Step 1 (ETL) failed: {e}")
            return

    states_to_train = args.states
    if not states_to_train:
        logging.warning("No states specified for training. Exiting workflow.")
        return

    logging.info(f"Step 2: Training for states: {', '.join(states_to_train)}")

    if args.parallel and len(states_to_train) > 1:
        logging.info(f"Running parallel training for {len(states_to_train)} states.")
        with ProcessPoolExecutor() as executor:
            executor.map(train.run_experiments, states_to_train)
    else:
        logging.info("Running sequential training.")
        for state_code in states_to_train:
            train.run_experiments(state=state_code)

    logging.info("Step 2 completed successfully.")
    end_time = time.time()
    logging.info(f"Workflow completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML pipeline orchestrator for dengue forecasting."
    )
    parser.add_argument(
        "--states",
        nargs="+",
        required=True,
        help="State abbreviations to train (e.g. CE SP RJ).",
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip the ETL step if data is already up to date.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train multiple states in parallel using separate processes.",
    )

    args = parser.parse_args()
    main(args)
