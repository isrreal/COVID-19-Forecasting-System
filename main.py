import asyncio
from src import predict
from src import data_processing
from src import train

async def main():

    data_processing.main_etl_pipeline()

    states_to_train = ["CE"]#, "SP", "RJ", "PE"]
    
    for state_code in states_to_train:
        train.run_experiments(state = state_code)
if __name__ == "__main__":
    asyncio.run(main())