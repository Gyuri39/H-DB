import streamlit as st
import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from .graph_scale import ReciprocalScale
from firebase_admin import firestore
import io
import xlsxwriter

DATA_DIR = "client/datas/"
MODEL_DIR = "client/models/"
MATERIAL_LIST = ["bcc-W",
					"bcc-Nb",
					"bcc-Fe",
					"fcc-Cu",
					"fcc-Ni",
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

DATA_TYPE_LIST = ["Raw data", "From fitted curve"]

HYDROGEN_DICT = {"H": "hydrogen atom", "D": "deuterium atom", "T": "tritium atom",
					"H2": "H2 molecule", "D2": "D2 molecule", "T2": "T2 molecule",
					"HD": "HD molecule", "DT": "DT molecule", "HT": "HT molecule",
					"else": "Combination, no information, etc."}	# TEMP(temperature) is regarded as the common information

HYDROGEN_GROUP = {"all isotope atoms": ["H", "D", "T"],
					"all isotope molecules": ["H2", "D2", "T2", "HD", "DT", "HT"]}

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
		elif isinstance(filelist[0], str):
			self.filelist = filelist
		else:
			raise TypeError(f"The element of filelist: {type(filelist)} must be a string representing a data ID.")

def filter_filelist(collection_name="datasets", material=None, hydrogen=None, attribute=None, method=None) -> FilterResult:
	db_client = firestore.client()
	collection_ref = db_client.collection(collection_name)

	def find_index(argument, listname):
		indices = [i for i, val in enumerate(listname) if val == argument]
		if len(indices) == 0:
			raise ValueError("No argument in the list")
		elif len(indices) == 1:
			return indices[0]
		else:
			raise ValueError("Two or more arguments in the list")
	def get_query(material_filter=None, hydrogen_filter=None, attribute_filter=None, method_filter=None):
		query = collection_ref
		if material_filter:
			query = query.where(filter=firestore.FieldFilter("material", "==", material_filter))
		if hydrogen_filter:
			if hydrogen_filter in HYDROGEN_DICT.keys():
				query = query.where(filter=firestore.FieldFilter("hydrogen", "==", hydrogen_filter))
			elif hydrogen_filter in HYDROGEN_GROUP.keys():
				query = query.where("hydrogen", "in", HYDROGEN_GROUP[hydrogen_filter])
			else:
				query = query
		if attribute_filter:
			query = query.where(filter=firestore.FieldFilter("attribute", "==", attribute_filter))
		if method_filter:
			query = query.where(filter=firestore.FieldFilter("method", "==", method_filter))
		return query
	
	if 'filtered_files' not in st.session_state:
		st.session_state.filterd_files = []
	if 'material_filter' not in st.session_state:
		st.session_state.material_filter = None
	if 'hydrogen_filter' not in st.session_state:
		st.session_state.hydrogen_filter = None
	if 'attribute_filter' not in st.session_state:
		st.session_state.attribute_filter = None
	if 'method_filter' not in st.session_state:
		st.session_state.method_filter = None

	if material:
		material_filter = st.selectbox("Filter by material", MATERIAL_LIST, index=find_index(material, MATERIAL_LIST))
	else:
		material_filter = st.selectbox("Filter by material", MATERIAL_LIST, index=None)
	hydrogen_filter_list = list(HYDROGEN_DICT.keys()) + list(HYDROGEN_GROUP.keys())
	if hydrogen:
		hydrogen_filter = st.selectbox("Filter by hydrogen", hydrogen_filter_list, index=find_index(hydrogen, hydrogen_filter_list))
	else:
		hydrogen_filter = st.selectbox("Filter by hydrogen", hydrogen_filter_list, index=None)
	if attribute:
		attribute_filter = st.selectbox("Filter by property", PROPERTY_DICT.keys(), index=find_index(attribute, PROPERTY_DICT.keys()))
	else:
		attribute_filter = st.selectbox("Filter by property", PROPERTY_DICT.keys(), index=None)
	if method:
		method_filter = st.selectbox("Filter by method", METHOD_LIST, index=find_index(method, METHOD_LIST))
	else:
		method_filter = st.selectbox("Filter by method", METHOD_LIST, index=None)

	if (material_filter != st.session_state.material_filter or
		hydrogen_filter != st.session_state.hydrogen_filter or
		attribute_filter != st.session_state.attribute_filter or
		method_filter != st.session_state.method_filter):
		
		st.session_state.material_filter = material_filter
		st.session_state.hydrogen_filter = hydrogen_filter
		st.session_state.attribute_filter = attribute_filter
		st.session_state.method_filter = method_filter


	query = get_query(st.session_state.material_filter, st.session_state.hydrogen_filter, st.session_state.attribute_filter, st.session_state.method_filter)
	docs = query.stream()
	filtered_files = []
	for doc in docs:
		doc_data = doc.to_dict()
		filtered_files.append(doc.id)
	st.session_state.filtered_files = filtered_files
	
	return FilterResult(material = material_filter, hydrogen = hydrogen_filter, attribute = attribute_filter, method = method_filter, filelist = filtered_files)

def convert_to_dataframe(st_file):
	suffix = Path(st_file.name).suffix
	if suffix == '.csv':
		df = pd.read_csv(st_file)
	elif suffix == '.txt':
		df = pd.read_table(st_file, sep=r'[ \t.]')
	elif suffix == 'xls' or 'xlsx':
		df = pd.read_excel(st_file)
	else:
		st.error("Critical error: invalid format of input data")
	return df

def save_csv_button(df: pd.DataFrame, use_container_width=False):
	csv = df.to_csv(index=False).encode('utf-8')
	st.download_button(
		label="Download in CSV",
		data=csv,
		file_name='data.csv',
		mime='text/csv',
		use_container_width=use_container_width
	)

def create_excel_file(dfs: list):
	# dfs must be a list of {'df': df, 'name1': 'filename', 'name2': 'user_defined_label'}
	name2_list = [item['name2'] for item in dfs]
	unique_name2 = set(name2_list)
	duplicate_name2 = {name for name in unique_name2 if name2_list.count(name) > 1}
	disable_name2 = len(duplicate_name2) > 0
	if disable_name2:
		options = ["data ID"]
		st.warning("Duplicate labels in selected data are detected")
	else:
		options = ["data ID", "label"]
	selected_option = st.radio("Choose the name of the individual sheet:", options, index=0, horizontal=True)
	
	buf = io.BytesIO()
	with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
		startcol = 0
		sheet_name = "compilation"
		worksheet = writer.book.add_worksheet(sheet_name)
		for item in dfs:
			df, name1, name2 = item['df'], item['name1'], item['name2']
			writer.sheets[sheet_name] = worksheet
			worksheet.write(0, startcol, name1)
			worksheet.write(1, startcol, name2)
			df.to_excel(writer, sheet_name=sheet_name, startcol=startcol, startrow=2, index=False)
			startcol += df.shape[1] + 1
		for item in dfs:
			df, name1, name2 = item['df'], item['name1'], item['name2']
			sheet_name_ind = name1 if selected_option == "data ID" else name2
			df.to_excel(writer, sheet_name=sheet_name_ind, index=False)
	buf.seek(0)
	st.download_button(label="Download Excel", data=buf, file_name="compilation.xlsx", mime="application/vnd.openxmlformat-officedocument.spreadsheetml.sheet")

def are_dataframes_equal(df1, df2, tolerance=0.05):	#'tolerance' relative tolerance and zero absolute tolerance
	common_cols = sorted(set(df1.columns) & set(df2.columns))
	df1_common = df1[common_cols].astype(float)
	df2_common = df2[common_cols].astype(float)
	df1_sorted = df1_common.sort_values(by=common_cols).reset_index(drop=True)
	df2_sorted = df2_common.sort_values(by=common_cols).reset_index(drop=True)
	if df1_sorted.shape != df2_sorted.shape:
		return False
	return np.allclose(df1_sorted, df2_sorted, rtol=tolerance, atol=0) 
