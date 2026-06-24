from pydantic import BaseModel, Field


class SummaryStats(BaseModel):
    total_notifications: int = Field(
        ..., description="Total dengue notifications in the database"
    )
    total_deaths: int = Field(..., description="Total deaths by dengue (outcome=2)")
    hospitalization_rate: float = Field(
        ..., description="Proportion of hospitalized cases"
    )
    mortality_rate: float = Field(..., description="Proportion of fatal cases")


class MunicipalityStats(BaseModel):
    municipality_code: int = Field(..., description="Municipality IBGE code")
    total_notifications: int = Field(
        ..., description="Total notifications for this municipality"
    )
    total_deaths: int = Field(..., description="Total deaths by dengue")
    hospitalization_rate: float = Field(
        ..., description="Proportion of hospitalized cases"
    )
    mortality_rate: float = Field(..., description="Proportion of fatal cases")


class MunicipalityNotification(BaseModel):
    municipality_code: int = Field(..., description="Municipality IBGE code")
    state_code: int = Field(..., description="State IBGE code")
    total_notifications: int = Field(..., description="Total notifications")


class MunicipalityMortality(BaseModel):
    municipality_code: int = Field(..., description="Municipality IBGE code")
    state_code: int = Field(..., description="State IBGE code")
    mortality_rate: float = Field(
        ..., description="Deaths divided by total notifications"
    )
    total_deaths: int = Field(..., description="Total deaths by dengue")
    total_notifications: int = Field(..., description="Total notifications")


class MunicipalityNotificationList(BaseModel):
    data: list[MunicipalityNotification]


class MunicipalityMortalityList(BaseModel):
    data: list[MunicipalityMortality]


class ChiSquareResult(BaseModel):
    test: str = Field(..., description="Name of the statistical test performed")
    null_hypothesis: str = Field(..., description="Statement of the null hypothesis")
    chi2_statistic: float = Field(..., description="Chi-square test statistic")
    p_value: float = Field(..., description="P-value of the test")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom")
    significance_level: float = Field(
        ..., description="Significance threshold used (e.g. 0.05)"
    )
    reject_null_hypothesis: bool = Field(
        ..., description="Whether the null hypothesis is rejected"
    )
    interpretation: str = Field(
        ..., description="Plain-language interpretation of the result"
    )
    contingency_table: dict[str, dict[str, int]] = Field(
        ..., description="Observed frequency contingency table"
    )
    expected_frequencies: list[list[float]] = Field(
        ..., description="Expected frequencies under independence"
    )


class ConfidenceInterval(BaseModel):
    metric: str = Field(
        ..., description="Metric name (e.g. daily_notifications, daily_deaths)"
    )
    mean: float = Field(..., description="Sample mean")
    lower: float = Field(..., description="Lower bound of the confidence interval")
    upper: float = Field(..., description="Upper bound of the confidence interval")
    n: int = Field(..., description="Sample size used in the calculation")
