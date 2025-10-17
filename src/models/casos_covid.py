from sqlalchemy import Column, BigInteger, Integer, String, Date, Float, Boolean
from sqlalchemy.orm import declarative_base
from datetime import date

ModelBase = declarative_base()

class CasoCovid(ModelBase):
    """
    Modelo ORM que representa uma linha de dados da tabela de casos de COVID.
    """
    __tablename__: str = 'casos_covid'

    id: int = Column(BigInteger, primary_key = True, autoincrement = True)
   
    datetime : date = Column(Date, index = True)
    state: str = Column(String(2), index = True)
    
    city: str = Column(String(255), nullable = True) 
    city_ibge_code: int = Column(Integer, nullable = True) 
    
    place_type: str = Column(String(50))
    
    last_available_confirmed: int = Column(BigInteger)
    new_confirmed: int = Column(Integer)
    
    last_available_deaths: int = Column(BigInteger)
    new_deaths: int = Column(Integer)

    estimated_population: int = Column(BigInteger, nullable = True)

    def __repr__(self) -> str:
        city_name = self.city or 'N/A (Estado)'
        return f"<CasoCovid(id={self.id}, date='{self.datetime}', state='{self.state}', city='{city_name}')>"