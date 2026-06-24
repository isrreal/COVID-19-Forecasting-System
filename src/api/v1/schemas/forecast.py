from pydantic import BaseModel, Field
from datetime import date as Date


class PredictionRequest(BaseModel):
    """Input sequence for a single-step prediction.

    Attributes:
        sequence: Ordered list of historical case values fed into the model
    """

    sequence: list[float] = Field(
        ..., description="Ordered list of historical case values fed into the model"
    )


class ForecastItem(BaseModel):
    """A single forecasted data point.

    Attributes:
        date: Forecasted date
        predicted_value: Predicted daily dengue case count
    """

    date: Date = Field(..., description="Forecasted date")
    predicted_value: float = Field(..., description="Predicted daily dengue case count")


class ForecastResponse(BaseModel):
    """Multi-step forecast for an entire state (aggregated).

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        forecast: Ordered list of daily forecasted values
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(
        ..., description="MLflow run ID of the model used for inference"
    )
    forecast: list[ForecastItem] = Field(
        ..., description="Ordered list of daily forecasted values"
    )


class ForecastResponseByMunicipality(BaseModel):
    """Multi-step forecast broken down by municipality within a state.

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        forecasts: Mapping of municipality IBGE code to its ordered list of forecasted values
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(
        ..., description="MLflow run ID of the model used for inference"
    )
    forecasts: dict[str, list[ForecastItem]] = Field(
        ...,
        description="Mapping of municipality IBGE code to its ordered list of forecasted values",
    )


class ForecastMunicipalityResponse(BaseModel):
    """Multi-step forecast for a specific municipality within a state.

    Attributes:
        state: State code (e.g. CE, SP)
        municipality_code: Municipality IBGE code
        model_run_id: MLflow run ID of the model used for inference
        forecast: Ordered list of daily forecasted values
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    municipality_code: int = Field(..., description="Municipality IBGE code")
    model_run_id: str = Field(
        ..., description="MLflow run ID of the model used for inference"
    )
    forecast: list[ForecastItem] = Field(
        ..., description="Ordered list of daily forecasted values"
    )


class ConfidenceForecastItem(BaseModel):
    """A single forecasted data point with confidence interval bounds.

    Attributes:
        date: Forecasted date
        predicted_mean: Mean predicted number of new confirmed cases
        lower_bound: Lower bound of the confidence interval
        upper_bound: Upper bound of the confidence interval
    """

    date: Date = Field(..., description="Forecasted date")
    predicted_mean: float = Field(
        ..., description="Mean predicted number of new confirmed cases"
    )
    lower_bound: float = Field(
        ..., description="Lower bound of the confidence interval"
    )
    upper_bound: float = Field(
        ..., description="Upper bound of the confidence interval"
    )


class ForecastConfidenceResponse(BaseModel):
    """Multi-step forecast with confidence intervals for an entire state.

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        confidence_level: Confidence level used (e.g. 0.95)
        forecast_with_confidence: Ordered list of forecasted values with confidence bounds
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(
        ..., description="MLflow run ID of the model used for inference"
    )
    confidence_level: float = Field(
        ..., description="Confidence level used (e.g. 0.95)"
    )
    forecast_with_confidence: list[ConfidenceForecastItem] = Field(
        ..., description="Ordered list of forecasted values with confidence bounds"
    )


class PredictionResponse(BaseModel):
    """Single-step prediction result.

    Attributes:
        state: State code (e.g. CE, SP)
        model_run_id: MLflow run ID of the model used for inference
        prediction: Predicted number of new confirmed cases for the next step
    """

    state: str = Field(..., description="State code (e.g. CE, SP)")
    model_run_id: str = Field(
        ..., description="MLflow run ID of the model used for inference"
    )
    prediction: float = Field(
        ..., description="Predicted number of new confirmed cases for the next step"
    )
