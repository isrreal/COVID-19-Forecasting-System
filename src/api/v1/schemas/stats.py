from pydantic import BaseModel


class SummaryStats(BaseModel):
    total_records: int
    total_confirmed: float
    total_deaths: float
    avg_new_confirmed_per_day: float
    avg_new_deaths_per_day: float


class CityStats(BaseModel):
    city: str
    total_confirmed: float
    total_deaths: float
    avg_new_confirmed: float
    avg_new_deaths: float


class CityConfirmed(BaseModel):
    city: str
    total_confirmed: float


class CityMortality(BaseModel):
    city: str
    state: str
    mortality_rate: float
    total_deaths: float
    total_confirmed: float


class CityConfirmedList(BaseModel):
    data: list[CityConfirmed]


class CityMortalityList(BaseModel):
    data: list[CityMortality]


class ChiSquareResult(BaseModel):
    test: str
    null_hypothesis: str
    chi2_statistic: float
    p_value: float
    degrees_of_freedom: int
    significance_level: float
    reject_null_hypothesis: bool
    interpretation: str
    contingency_table: dict[str, dict[int, int]]
    expected_frequencies: list[list[float]]


class ConfidenceInterval(BaseModel):
    metric: str
    mean: float
    lower: float
    upper: float
    n: int
