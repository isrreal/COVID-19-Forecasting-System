from datetime import date

from sqlalchemy import BigInteger, Date, Integer, SmallInteger, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ModelBase(DeclarativeBase):
    pass


class CasoDengue(ModelBase):
    __tablename__ = "casos_dengue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    notification_date: Mapped[date] = mapped_column(Date, index=True)
    epidemiological_week: Mapped[str] = mapped_column(String(6), index=True)
    year: Mapped[int] = mapped_column(SmallInteger, index=True)

    state_ibge_code: Mapped[int] = mapped_column(SmallInteger, index=True)
    municipality_ibge_code: Mapped[int] = mapped_column(Integer, index=True)

    # 10=dengue, 11=dengue with alarm signs, 12=severe dengue, 8=discarded
    final_classification: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True
    )

    # 1=recovery, 2=death by dengue, 3=death by other cause, 9=unknown
    outcome: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # 1=yes, 2=no, 9=unknown
    hospitalized: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # 1–4, None when not identified
    serotype: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    death_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # M, F, None (unknown)
    sex: Mapped[str | None] = mapped_column(String(1), nullable=True)

    # SINAN encoding: first digit = unit (4=years, 3=months, 2=days), remaining digits = value
    age_encoded: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<CasoDengue(id={self.id}, date='{self.notification_date}', "
            f"state={self.state_ibge_code}, classification={self.final_classification})>"
        )
