from pydantic import BaseModel, Field
from datetime import date as Date


class PredictionRequest(BaseModel):
    """Input sequence for a single-step prediction.

    Attributes:
        sequence: Ordered list of historical case values fed into the model
    """

    sequence: list[float] = Field(..., description="Ordered list of historical case values fed into the model")


class ForecastItem(BaseModel):
    """A single forecasted data point.

    Attributes:
        date: Forecasted date
        predicted_value: Predicted number of new confirmed cases
    """

    date: Date = Field(..., description="Forecasted date")
    predicted_value: float = Field(..., description="Predicted number of new confirmed cases")


class ForecastResponse(BaseModel):
    """Multi-step forecast for an entire state (aggregated).

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        forecast: Ordered list of daily forecasted values
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(..., description="MLflow run ID of the model used for inference")
    forecast: list[ForecastItem] = Field(..., description="Ordered list of daily forecasted values")


class ForecastResponseByCity(BaseModel):
    """Multi-step forecast broken down by city within a state.

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        forecasts: Mapping of city name to its ordered list of forecasted values
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(..., description="MLflow run ID of the model used for inference")
    forecasts: dict[str, list[ForecastItem]] = Field(..., description="Mapping of city name to its ordered list of forecasted values")


class PredictionResponse(BaseModel):
    """Single-step prediction result.

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        prediction: Predicted number of new confirmed cases for the next step
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(..., description="MLflow run ID of the model used for inference")
    prediction: float = Field(..., description="Predicted number of new confirmed cases for the next step")
