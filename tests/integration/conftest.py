import os
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.models.casos_covid import CasoCovid, ModelBase

TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_DATABASE_URL)
    ModelBase.metadata.create_all(bind=engine)
    yield engine
    ModelBase.metadata.drop_all(bind=engine)


@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    db = Session(bind=connection)
    yield db
    db.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_data(session):
    records = [
        CasoCovid(
            datetime=date(2021, 1, 1),
            state="CE",
            city="fortaleza",
            place_type="city",
            last_available_confirmed=1000,
            new_confirmed=100,
            last_available_deaths=50,
            new_deaths=5,
        ),
        CasoCovid(
            datetime=date(2021, 1, 2),
            state="CE",
            city="fortaleza",
            place_type="city",
            last_available_confirmed=1100,
            new_confirmed=100,
            last_available_deaths=55,
            new_deaths=5,
        ),
        CasoCovid(
            datetime=date(2021, 1, 1),
            state="SP",
            city="sao paulo",
            place_type="city",
            last_available_confirmed=5000,
            new_confirmed=500,
            last_available_deaths=200,
            new_deaths=20,
        ),
        CasoCovid(
            datetime=date(2021, 1, 1),
            state="CE",
            city="sobral",
            place_type="city",
            last_available_confirmed=200,
            new_confirmed=20,
            last_available_deaths=2,
            new_deaths=0,
        ),
    ]
    session.add_all(records)
    session.flush()
    return records
