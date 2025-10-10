import asyncio
from src import data_processing
from src import train
from database import get_async_session
from src.models.casos_covid import CasoCovid
from sqlalchemy import select

async def consulta_teste():
    print("\nIniciando consulta de teste assíncrona...")
    async with get_async_session() as session:
        query = select(CasoCovid).limit(5) 
        result = await session.execute(query)
        casos = result.scalars().all()

        print(f"Encontrados {len(casos)} casos na consulta de teste:")
        for caso in casos:
            print(f"  - Data: {caso.datetime}, Estado: {caso.state}, Cidade: {caso.city or 'N/A'}, Novos Casos: {caso.new_confirmed}, Novas Mortes: {caso.new_deaths}, Total de Casos: {caso.last_available_confirmed}")
    print("Consulta de teste finalizada.")

async def main():
    data_processing.etl_pipeline()
    
    train.train_model()

if __name__ == "__main__":
    asyncio.run(main())

