from datetime import date
from typing import Optional

from sqlalchemy import BigInteger, Integer, String, Date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ModelBase(DeclarativeBase):
    pass


class CasoCovid(ModelBase):
    """ORM model representing a row of COVID case data."""

    __tablename__: str = "casos_covid"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    datetime: Mapped[date] = mapped_column(Date, index=True)
    state: Mapped[str] = mapped_column(String(2), index=True)

    city: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    city_ibge_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    place_type: Mapped[str] = mapped_column(String(50))

    last_available_confirmed: Mapped[int] = mapped_column(BigInteger)
    new_confirmed: Mapped[int] = mapped_column(Integer)

    last_available_deaths: Mapped[int] = mapped_column(BigInteger)
    new_deaths: Mapped[int] = mapped_column(Integer)

    estimated_population: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )

    def __repr__(self) -> str:
        city_name = self.city or "N/A (Estado)"
        return f"<CasoCovid(id={self.id}, date='{self.datetime}', state='{self.state}', city='{city_name}')>"
