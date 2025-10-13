import asyncio
from src import predict
import os
import mlflow

async def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    predict.predict_next_day(run_id = predict.BEST_RUN_ID)


if __name__ == "__main__":
    asyncio.run(main())