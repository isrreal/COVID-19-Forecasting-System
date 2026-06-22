from src.api.v1.services import stats_service


def test_get_summary_stats_counts_all_records(session, sample_data):
    result = stats_service.get_summary_stats(session)
    assert result["total_records"] == 4
    assert result["total_confirmed"] > 0
    assert result["total_deaths"] > 0


def test_get_summary_stats_empty_db(session):
    result = stats_service.get_summary_stats(session)
    assert result["total_records"] == 0
    assert result["total_confirmed"] == 0.0


def test_get_city_stats_returns_correct_city(session, sample_data):
    result = stats_service.get_city_stats("Fortaleza", "CE", session)
    assert result is not None
    assert result["city"] == "fortaleza"
    assert result["total_confirmed"] == 2100.0
    assert result["total_deaths"] == 105.0


def test_get_city_stats_normalizes_accents(session, sample_data):
    result = stats_service.get_city_stats("São Paulo", "SP", session)
    assert result is not None
    assert result["city"] == "sao paulo"


def test_get_city_stats_returns_none_for_unknown_city(session, sample_data):
    result = stats_service.get_city_stats("Cidade Inexistente", "CE", session)
    assert result is None


def test_get_top_cities_returns_sorted_by_confirmed(session, sample_data):
    result = stats_service.get_top_cities(10, session)
    assert isinstance(result, list)
    assert result[0]["city"] == "sao paulo"
    assert result[0]["total_confirmed"] > result[1]["total_confirmed"]


def test_get_top_cities_respects_limit(session, sample_data):
    result = stats_service.get_top_cities(1, session)
    assert len(result) == 1


def test_get_most_deadly_cities_sorted_by_mortality(session, sample_data):
    result = stats_service.get_most_deadly_cities(10, session)
    assert isinstance(result, list)
    assert len(result) > 0
    rates = [r["mortality_rate"] for r in result]
    assert rates == sorted(rates, reverse=True)


def test_get_least_affected_cities_sorted_ascending(session, sample_data):
    result = stats_service.get_least_affected_cities(10, session)
    assert isinstance(result, list)
    rates = [r["mortality_rate"] for r in result]
    assert rates == sorted(rates)


def test_confidence_interval_cases_bounds(session, sample_data):
    result = stats_service.get_confidence_interval_cases(session)
    assert result["lower"] <= result["mean"] <= result["upper"]
    assert result["metric"] == "new_confirmed"
    assert result["n"] == 4


def test_confidence_interval_deaths_bounds(session, sample_data):
    result = stats_service.get_confidence_interval_deaths(session)
    assert result["lower"] <= result["mean"] <= result["upper"]
    assert result["metric"] == "new_deaths"


def test_chi_square_returns_expected_fields(session, sample_data):
    result = stats_service.chi_square_state_deaths(session)
    assert "chi2_statistic" in result
    assert "p_value" in result
    assert "reject_null_hypothesis" in result
    assert isinstance(result["reject_null_hypothesis"], bool)
