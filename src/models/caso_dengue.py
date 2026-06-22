from datetime import date

from sqlalchemy import BigInteger, Date, Integer, SmallInteger, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ModelBase(DeclarativeBase):
    pass


class CasoDengue(ModelBase):
    __tablename__ = "casos_dengue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    dt_notific: Mapped[date] = mapped_column(Date, index=True)
    sem_not: Mapped[str] = mapped_column(String(6), index=True)
    nu_ano: Mapped[int] = mapped_column(SmallInteger, index=True)

    sg_uf_not: Mapped[int] = mapped_column(SmallInteger, index=True)
    id_municip: Mapped[int] = mapped_column(Integer, index=True)

    # 10=dengue, 11=dengue with alarm signs, 12=severe dengue, 8=discarded
    classi_fin: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # 1=recovery, 2=death by dengue, 3=death by other cause, 9=unknown
    evolucao: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # 1=yes, 2=no, 9=unknown
    hospitaliz: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # 1–4, None when not identified
    sorotipo: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    dt_obito: Mapped[date | None] = mapped_column(Date, nullable=True)

    # M, F, I (unknown)
    cs_sexo: Mapped[str | None] = mapped_column(String(1), nullable=True)

    # SINAN encoding: first digit = unit (4=years, 3=months, 2=days), remaining digits = value
    nu_idade_n: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<CasoDengue(id={self.id}, data='{self.dt_notific}', "
            f"uf={self.sg_uf_not}, classi={self.classi_fin})>"
        )
