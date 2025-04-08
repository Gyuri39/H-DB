import streamlit as st
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import pickle
from pathlib import Path
import fnmatch
import numpy as np
from firebase_admin import firestore
from io import BytesIO
import base64

@dataclass
class DataWithDescription:
	material: str
	hydrogen: str
	attribute: str
	method: str
	date: str
	uploader: str
	data: pd.DataFrame
	data_type: str
	information_source: str
	who_measured: str
	descript_purity: str
	descript_pretreatment: str
	descript_methoddetail: str
	descript_else: str
	who_verified: dict
	pdf_flag: bool

def make_DWD(material, hydrogen, attribute, method, uploader, data, data_type, information_source, who_measured, descript_purity, descript_pretreatment, descript_methoddetail, descript_else) -> DataWithDescription:
	now = datetime.now()
	date = now.strftime("%Y-%m-%d_%H-%M-%S")
	return DataWithDescription(material, hydrogen, attribute, method, date, uploader, data, data_type, information_source, who_measured, descript_purity, descript_pretreatment, descript_methoddetail, descript_else, who_verified={}, pdf_flag=False)

def modify_DWD(DWD: DataWithDescription, which2mod, how2mod) -> DataWithDescription:
	if hasattr(DWD, which2mod):
		setattr(DWD, which2mod, how2mod)
	else:
		raise NotImplementedError(f"Error: '{which2mod}' is not an attribute of DWD")

def serialized_dataframe(df: pd.DataFrame) -> str:
	buffer = BytesIO()
	pickle.dump(df, buffer)
	return base64.b64encode(buffer.getvalue()).decode("utf-8")

def deserialize_dataframe(encoded_str: str) -> pd.DataFrame:
	decoded_bytes = base64.b64decode(encoded_str)
	buffer = BytesIO(decoded_bytes)
	return pickle.load(buffer)
	
def save_DWD(DWD: DataWithDescription, collection_name="datasets"):
	db_client = firestore.client()
	collection_ref = db_client.collection(collection_name)
	serialized_data = serialized_dataframe(DWD.data)
	
	if DWD.material == None or DWD.hydrogen == None or DWD.attribute == None or DWD.method == None:
		st.warning("Necessary attributes are not properly implemented.")
		return False
	query = (
		collection_ref
		.where(filter=firestore.FieldFilter("material", "==", DWD.material))
		.where(filter=firestore.FieldFilter("hydrogen", "==", DWD.hydrogen))
		.where(filter=firestore.FieldFilter("attribute", "==", DWD.attribute))
		.where(filter=firestore.FieldFilter("method", "==", DWD.method))
	)
	docs = query.stream()
	existing_docs = list(docs)

	if existing_docs:
		for doc in docs:
			existing_data = doc.to_dict()
			existing_serialized_data = existing_data["data"]
			existing_df = deserialize_dataframe(existing_serialized_data)
			if are_dataframes_equal(DWD.data, existing_df):
				st.warning(f"Duplicate data found: {doc.id}")
				return False

	#doc_id = f"{DWD.material}_{DWD.hydrogen}_{DWD.attribute}_{DWD.method}_{DWD.date}"
	doc_id = f"{DWD.date}.dat"
	collection_ref.document(doc_id).set({
		"material": DWD.material,
		"hydrogen": DWD.hydrogen,
		"attribute": DWD.attribute,
		"method": DWD.method,
		"date": DWD.date,
		"data_type": DWD.data_type,
		"uploader": DWD.uploader,
		"information_source": DWD.information_source,
		"who_measured": DWD.who_measured,
		"descript_purity": DWD.descript_purity,
		"descript_pretreatment": DWD.descript_pretreatment,
		"descript_methoddetail": DWD.descript_methoddetail,
		"descript_else": DWD.descript_else,
		"who_verified": DWD.who_verified,
		"data": serialized_data,
		"pdf_flag": DWD.pdf_flag
	})
	st.success(f"The file {doc_id} is saved on the server")
	return True

def load_DWD(doc_id, collection_name="datasets"):
	db_client = firestore.client()
	collection_ref = db_client.collection(collection_name)
	doc_ref = collection_ref.document(doc_id)
	doc = doc_ref.get()
	if not doc.exists:
		st.error(f"No document found with ID: {doc_id}")
	data = doc.to_dict()

	df = deserialize_dataframe(data["data"])
	DWD = DataWithDescription(
		material = data["material"],
		hydrogen = data["hydrogen"],
		attribute = data["attribute"],
		method = data["method"],
		date = data["date"],
		uploader = data["uploader"],
		data = df,
		data_type = data["data_type"],
		information_source = data["information_source"],
		who_measured = data["who_measured"],
		descript_purity = data["descript_purity"],
		descript_pretreatment = data["descript_pretreatment"],
		descript_methoddetail = data["descript_methoddetail"],
		descript_else = data["descript_else"],
		who_verified = data["who_verified"],
		pdf_flag = data["pdf_flag"]
	)
	return DWD

def info_DWD(DWD: DataWithDescription, info: str):
	if info == 'data':
		return DWD.data
	elif info == 'who_verified':
		return DWD.who_verified
	elif info == 'pdf_flag':
		return DWD.pdf_flag
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
