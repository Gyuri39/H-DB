import streamlit as st
import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from .graph_scale import ReciprocalScale

DATA_DIR = "client/datas/"
MODEL_DIR = "client/models/"
MATERIAL_LIST = ["bcc-W",
					"bcc-Nb",
					"bcc-Fe",
					"fcc-Cu",
					"hcp-Zr",
					"SS304",
					"SS316",
					"H-gas",
					"H-liquid",
					"H-solid"
					]
PROPERTY_DICT = {"Diffusivity": "diffusion coefficient in m<sup>2</sup> s<sup>-1</sup>",
					"Solubility": "solubility constant in mol m<sup>-3</sup> Pa<sup>-0.5</sup>",
					"Permeability": "permeability constant in mol m<sup>-1</sup> s<sup>-1</sup> Pa<sup>-0.5</sup>",
					"Heat capacity": "Cp in J K<sup>-1</sup> mol<sup>-1</sup>"
					}
METHOD_LIST = ["Experiment", "Simulation", "Theory/Model"]

KEY_PROPERTY = {"H": "hydrogen atom", "D": "deuterium atom", "T": "tritium atom",
					"H2": "H2 molecule", "D2": "D2 molecule", "T2": "T2 molecule",
					"HD": "HD molecule", "DT": "DT molecule", "HT": "HT molecule",
					"else": "Combination, no information, etc."}	# TEMP(temperature) is regarded as the common information

VALID_DATA_FORMAT = ['csv', 'txt', 'xls', 'xlsx']

@dataclass
class FilterResult():
	material: str = None
	hydrogen: str = None
	attribute: str = None
	method: str = None
	filelist: list = None

	def updateFilelist(self, filelist: list):
		if not filelist:
			raise KeyError("Empty updateFilelist")
		#elif isinstance(filelist[0], file):
		#	self.filelist = filelist
		elif isinstance(filelist[0], str):
			path = Path(DATA_DIR)
			all_files = list(path.glob("*.dat"))
			self.filelist = []
			for file in list(path.glob("*.dat")):
				if file.name in filelist:
					self.filelist.append(file)
		else:
			raise TypeError(f"The element of filelist: {type(filelist)} must has a type of either file or string.")

def filter_filelist(path=DATA_DIR, material=None, hydrogen=None, attribute=None, method=None) -> FilterResult:
	def find_index(argument, listname):
		indices = [i for i, val in enumerate(listname) if val == argument]
		if len(indices) == 0:
			raise ValueError("No argument in the list")
		elif len(indices) == 1:
			return indices[0]
		else:
			raise ValueError("Two or more arguments in the list")

	if material:
		material_filter = st.selectbox("Filter by material", MATERIAL_LIST, index=find_index(material, MATERIAL_LIST))
	else:
		material_filter = st.selectbox("Filter by material", MATERIAL_LIST, index=None)
	if hydrogen:
		hydrogen_filter = st.selectbox("Filter by hydrogen", KEY_PROPERTY.keys(), index=find_index(hydrogen, KEY_PROPERTY.keys()))
	else:
		hydrogen_filter = st.selectbox("Filter by hydrogen", KEY_PROPERTY.keys(), index=None)
	if attribute:
		attribute_filter = st.selectbox("Filter by property", PROPERTY_DICT.keys(), index=find_index(attribute, PROPERTY_DICT.keys()))
	else:
		attribute_filter = st.selectbox("Filter by property", PROPERTY_DICT.keys(), index=None)
	if method:
		method_filter = st.selectbox("Filter by method", METHOD_LIST, index=find_index(method, METHOD_LIST))
	else:
		method_filter = st.selectbox("Filter by method", METHOD_LIST, index=None)
	all_files = list(Path(path).glob("*.dat"))
	filtered_files = [
		file for file in all_files if
			(not material_filter or file.name.startswith(material_filter)) and
			(not hydrogen_filter or f"_{hydrogen_filter}_" in file.name) and
			(not attribute_filter or f"_{attribute_filter}_" in file.name) and
			(not method_filter or f"_{method_filter}_" in file.name)
	]
	st.write(f"Found {len(filtered_files)} files matching the criteria.")
	return FilterResult(material = material_filter, hydrogen = hydrogen_filter, attribute = attribute_filter, method = method_filter, filelist = filtered_files)

def convert_to_dataframe(st_file):
	suffix = Path(st_file.name).suffix
	if suffix == '.csv':
		df = pd.read_csv(st_file)
	elif suffix == '.txt':
		df = pd.read_read-table(st_file, sep=r'[ \t.]')
	elif suffix == 'xls' or 'xlsx':
		df = pd.read_excel(st_file)
	else:
		st.error("Critical error: invalid format of input data")
	return df

def save_csv_button(df: pd.DataFrame):
	csv = df.to_csv(index=False).encode('utf-8')
	st.download_button(
		label="Download in CSV",
		data=csv,
		file_name='data.csv',
		mime='text/csv'
	)

def are_dataframes_equal(df1, df2, tolerance=0.05):	#'tolerance' relative tolerance and zero absolute tolerance
	common_cols = sorted(set(df1.columns) & set(df2.columns))
	df1_common = df1[common_cols].astype(float)
	df2_common = df2[common_cols].astype(float)
	df1_sorted = df1_common.sort_values(by=common_cols).reset_index(drop=True)
	df2_sorted = df2_common.sort_values(by=common_cols).reset_index(drop=True)
	if df1_sorted.shape != df2_sorted.shape:
		return False
	return np.allclose(df1_sorted, df2_sorted, rtol=tolerance, atol=0)
