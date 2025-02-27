import streamlit as st
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import pickle
from pathlib import Path
import fnmatch
import numpy as np

@dataclass
class DataWithDescription:
	material: str
	hydrogen: str
	attribute: str
	method: str
	date: str
	uploader: str
	data: pd.DataFrame
	information_source: str
	who_measured: str
	descript_purity: str
	descript_pretreatment: str
	descript_methoddetail: str
	descript_else: str
	thumbsup: dict
	thumbsdown: dict

def make_DWD(material, hydrogen, attribute, method, uploader, data, information_source, who_measured, descript_purity, descript_pretreatment, descript_methoddetail, descript_else, thumbsup = {}, thumbsdown = {}) -> DataWithDescription:
	now = datetime.now()
	date = now.strftime("%Y-%m-%d_%H-%M-%S")
	return DataWithDescription(material, hydrogen, attribute, method, date, uploader, data, information_source, who_measured, descript_purity, descript_pretreatment, descript_methoddetail, descript_else, thumbsup, thumbsdown)

def modify_DWD(DWD: DataWithDescription, which2mod, how2mod) -> DataWithDescription:
	if hasattr(DWD, which2mod):
		setattr(DWD, which2mod, how2mod)
	else:
		raise NotImplementedError(f"Error: '{which2mod}' is not an attribute of DWD")

def save_DWD(DWD: DataWithDescription, file_path):
	dataset_path = Path(file_path)
	search_pattern = f"{DWD.material}_{DWD.hydrogen}_{DWD.attribute}_{DWD.method}_*.dat"
	for file_path in dataset_path.glob("*.dat"):
		if fnmatch.fnmatch(file_path.name, search_pattern):
			with file_path.open('rb') as f:
				existing_dwd = pickle.load(f)
			if are_dataframes_equal(DWD.data, existing_dwd.data):
				st.warning(f"Duplicate found: {file_path.name}")
				return False

	new_filename = f"{DWD.material}_{DWD.hydrogen}_{DWD.attribute}_{DWD.method}_{DWD.date}.dat"
	with open(dataset_path / new_filename, "wb") as f:
		pickle.dump(DWD, f)
		st.success(f"The file {new_filename} is saved on the server.")
	return True

def load_DWD(file_path):
	with open(file_path, "rb") as f:
		return pickle.load(f)

def info_DWD(DWD: DataWithDescription, info: str):
	if info == 'data':
		return DWD.data
	elif info == 'thumbsup':
		return DWD.thumbsup
	elif info == 'thumbsdown':
		return DWD.thumbsdown
	elif info in DWD.__dataclass_fields__:
		return str(getattr(DWD, info))
	else:
		raise ValueError(f"'{info}' is not a valid attribute of DataWithDescription.")



def are_dataframes_equal(df1, df2, tolerance=0.05): #'tolerance' relative tolerance and zero absolute tolerance
	common_cols = sorted(set(df1.columns) & set(df2.columns))
	df1_common = df1[common_cols].astype(float)
	df2_common = df2[common_cols].astype(float)
	df1_sorted = df1_common.sort_values(by=common_cols).reset_index(drop=True)
	df2_sorted = df2_common.sort_values(by=common_cols).reset_index(drop=True)
	if df1_sorted.shape != df2_sorted.shape:
		return False
	return np.allclose(df1_sorted, df2_sorted, rtol=tolerance, atol=0)
