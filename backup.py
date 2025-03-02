import math
import numpy as np
import pandas as pd
import sqlalchemy
from matplotlib import pyplot as plt
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


class Transaction:
	def __init__(self):
		self.session = Session()

	def __enter__(self):
		return self.session

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type is None:
			try:
				self.session.commit()
			except:
				self.session.rollback()
				raise
		else:
			self.session.rollback()
			raise
		self.session.close()


with Transaction() as transaction:
	transaction.query(SQLTable)


def process_item(column: list[str | int | np.float64]) -> list[str | int | float] | None:
	"""
	params: column: list[str|np.float_]
	return: list[str|np.float_] | None
	None for drop
	"""
	assert column[0], "Country name cannot be empty"
	assert column[1], "Year cannot be empty"
	assert column[2], "Total Energy Consumption (TWh) cannot be empty"

	column[1] = int(column[1])

	if not math.isnan(column[3]) and all([not math.isnan(item) for item in column[6:]]):
		return column

	if math.isnan(column[3]):
		if all([not math.isnan(item) for item in column[6:]]):
			column[3] = sum(column[6:])
			return column
		else:
			# There are over two NaN values in the dataset could fix.
			return None

	nan_index = None
	accumulated_value = 0
	for i, item in enumerate(column[6:], start=6):
		if math.isnan(item):
			if nan_index is not None:
				break
			nan_index = i
		else:
			accumulated_value += item
	else:
		column[nan_index] = column[3] - accumulated_value
		return column

	return None


def process_data(df_csv: pd.DataFrame) -> list[list[str | int | float]]:
	"""
	params: df_csv: pd.DataFrame
	return: list[list[str|int|np.float_]]
	"""
	processed_data = []
	for i in range(len(df_csv["Country"])):
		temp = []
		for row_name in df_csv:
			temp.append(df_csv[row_name][i])
		processed_item = process_item(temp)
		if processed_item is not None:
			processed_data.append(processed_item)

	interpolated_data = []
	for i, item in enumerate(processed_data):
		if not math.isnan(item[4]):
			interpolated_data.append(item)
			continue
		left_ptr = i - 1
		right_ptr = i + 1
		while left_ptr >= 0:
			if not math.isnan(processed_data[left_ptr][4]):
				break
			left_ptr -= 1
		else:
			left_ptr = None

		while right_ptr < len(processed_data):
			if not math.isnan(processed_data[right_ptr][4]):
				break
			right_ptr += 1
		else:
			right_ptr = None

		if left_ptr is None and right_ptr is None:
			# This is not linear interpolation
			continue

		if left_ptr is None:
			while right_ptr < len(processed_data):
				if not math.isnan(processed_data[right_ptr][4]):
					if left_ptr is not None:
						break
					left_ptr = right_ptr
				right_ptr += 1
			else:
				continue

		if right_ptr is None:
			while left_ptr >= 0:
				if not math.isnan(processed_data[left_ptr][4]):
					if right_ptr is not None:
						break
					right_ptr = left_ptr
				left_ptr -= 1
			else:
				continue

		assert left_ptr is not None and right_ptr is not None
		assert left_ptr < right_ptr

		k = (processed_data[right_ptr][4] + processed_data[left_ptr][4]) / (processed_data[right_ptr][1] - processed_data[left_ptr][1])
		item[4] = int(k * (item[1] - processed_data[left_ptr][1]) + processed_data[left_ptr][4])
		interpolated_data.append(item)
	return interpolated_data




if __name__ == "__main__":
	df_csv = pd.read_csv('Renewable_Energy_Adoption.csv')
	processed_data = process_data(df_csv)

	with Transaction() as transaction:
		for row in processed_data:
			transaction.add(SQLTable(Country=row[0], Year=row[1], Total_Energy_Consumption=row[2], Renewable_Energy=row[3],
									 Government_Investment=row[4], Emissions_Reduction=row[5], Solar=row[6], Wind=row[7],
									 Hydro=row[8], Geothermal=row[9], Biomass=row[10], Other_Renewables=row[11]))