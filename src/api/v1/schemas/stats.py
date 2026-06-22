from pydantic import BaseModel, Field


class SummaryStats(BaseModel):
    """Aggregated statistics across the entire dataset.

    Attributes:
        total_records: Total number of records in the database
        total_confirmed: Cumulative confirmed cases
        total_deaths: Cumulative deaths
        avg_new_confirmed_per_day: Average daily new confirmed cases
        avg_new_deaths_per_day: Average daily new deaths
    """

    total_records: int = Field(..., description="Total number of records in the database")
    total_confirmed: float = Field(..., description="Cumulative confirmed cases")
    total_deaths: float = Field(..., description="Cumulative deaths")
    avg_new_confirmed_per_day: float = Field(..., description="Average daily new confirmed cases")
    avg_new_deaths_per_day: float = Field(..., description="Average daily new deaths")


class CityStats(BaseModel):
    """Aggregated statistics for a specific city.

    Attributes:
        city: City name
        total_confirmed: Cumulative confirmed cases in the city
        total_deaths: Cumulative deaths in the city
        avg_new_confirmed: Average daily new confirmed cases
        avg_new_deaths: Average daily new deaths
    """

    city: str = Field(..., description="City name")
    total_confirmed: float = Field(..., description="Cumulative confirmed cases in the city")
    total_deaths: float = Field(..., description="Cumulative deaths in the city")
    avg_new_confirmed: float = Field(..., description="Average daily new confirmed cases")
    avg_new_deaths: float = Field(..., description="Average daily new deaths")


class CityConfirmed(BaseModel):
    """City with its total confirmed case count.

    Attributes:
        city: City name
        total_confirmed: Cumulative confirmed cases
    """

    city: str = Field(..., description="City name")
    total_confirmed: float = Field(..., description="Cumulative confirmed cases")


class CityMortality(BaseModel):
    """City mortality statistics including rate and totals.

    Attributes:
        city: City name
        state: State code (e.g. CE, SP)
        mortality_rate: Deaths divided by confirmed cases
        total_deaths: Cumulative deaths
        total_confirmed: Cumulative confirmed cases
    """

    city: str = Field(..., description="City name")
    state: str = Field(..., description="State code (e.g. CE, SP)")
    mortality_rate: float = Field(..., description="Deaths divided by confirmed cases")
    total_deaths: float = Field(..., description="Cumulative deaths")
    total_confirmed: float = Field(..., description="Cumulative confirmed cases")


class CityConfirmedList(BaseModel):
    """List of cities with their confirmed case counts.

    Attributes:
        data: List of cities with cumulative confirmed cases
    """

    data: list[CityConfirmed]


class CityMortalityList(BaseModel):
    """List of cities with their mortality statistics.

    Attributes:
        data: List of cities with mortality statistics
    """

    data: list[CityMortality]


class ChiSquareResult(BaseModel):
    """Result of a chi-square independence test between state and death occurrence.

    Attributes:
        test: Name of the statistical test performed
        null_hypothesis: Statement of the null hypothesis
        chi2_statistic: Chi-square test statistic
        p_value: P-value of the test
        degrees_of_freedom: Degrees of freedom
        significance_level: Significance threshold used (e.g. 0.05)
        reject_null_hypothesis: Whether the null hypothesis is rejected
        interpretation: Plain-language interpretation of the result
        contingency_table: Observed frequency contingency table
        expected_frequencies: Expected frequencies under independence
    """

    test: str = Field(..., description="Name of the statistical test performed")
    null_hypothesis: str = Field(..., description="Statement of the null hypothesis")
    chi2_statistic: float = Field(..., description="Chi-square test statistic")
    p_value: float = Field(..., description="P-value of the test")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom")
    significance_level: float = Field(..., description="Significance threshold used (e.g. 0.05)")
    reject_null_hypothesis: bool = Field(..., description="Whether the null hypothesis is rejected")
    interpretation: str = Field(..., description="Plain-language interpretation of the result")
    contingency_table: dict[str, dict[str, int]] = Field(..., description="Observed frequency contingency table")
    expected_frequencies: list[list[float]] = Field(..., description="Expected frequencies under independence")


class ConfidenceInterval(BaseModel):
    """Confidence interval for a daily metric mean.

    Attributes:
        metric: Metric name (e.g. new_confirmed, new_deaths)
        mean: Sample mean
        lower: Lower bound of the confidence interval
        upper: Upper bound of the confidence interval
        n: Sample size used in the calculation
    """

    metric: str = Field(..., description="Metric name (e.g. new_confirmed, new_deaths)")
    mean: float = Field(..., description="Sample mean")
    lower: float = Field(..., description="Lower bound of the confidence interval")
    upper: float = Field(..., description="Upper bound of the confidence interval")
    n: int = Field(..., description="Sample size used in the calculation")
