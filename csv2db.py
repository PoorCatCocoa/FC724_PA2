import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///FC724_PA2.db')
Base = sqlalchemy.orm.declarative_base()
Session = sessionmaker(bind=engine)


class SQLTable(Base):
	__tablename__ = 'REA'

	id = Column(Integer, primary_key=True)
	Country = Column(String)
	Year = Column(Integer)
	Total_Energy_Consumption = Column(Float, name='Total Energy Consumption (TWh)')
	Renewable_Energy = Column(Float, name='Renewable Energy (%)')
	Government_Investment = Column(Float, name='Government Investment (Million USD)')
	Emissions_Reduction = Column(Float, name='Emissions Reduction (%)')
	Solar = Column(Float)
	Wind = Column(Float)
	Hydro = Column(Float)
	Geothermal = Column(Float)
	Biomass = Column(Float)
	Other_Renewables = Column(Float, name='Other Renewables')

Base.metadata.create_all(engine)
