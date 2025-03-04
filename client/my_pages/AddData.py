import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from utils import DATA_DIR, MATERIAL_LIST, PROPERTY_DICT, METHOD_LIST, KEY_PROPERTY, VALID_DATA_FORMAT, convert_to_dataframe
from server.data.db_handler import make_DWD, save_DWD
from server.data.utils import candidate_columns

def createPage():
	st.title("Add data")
	con11, con12 = st.columns([0.5, 0.5])
	uploaded_file = st.file_uploader("Choose a file to upload", type = VALID_DATA_FORMAT, accept_multiple_files = False)
	to_suggest = None
	save_button = None
	if uploaded_file:
		df = convert_to_dataframe(uploaded_file)
	with con12:
		if uploaded_file:
			material = st.selectbox("*Material", MATERIAL_LIST, index=None)
			hydrogen = st.selectbox("*Form of Hydrogen", KEY_PROPERTY.keys(), index=None)
			if hydrogen:
				st.html(KEY_PROPERTY[hydrogen])
			attribute = st.selectbox("*Property", PROPERTY_DICT.keys(), index=None)
			if attribute:
				st.html(PROPERTY_DICT[attribute])
			method = st.selectbox("*Method", METHOD_LIST, index=None)
			uploader = st.session_state.name
			information_source = st.text_input("*Information source by DOI, web address, thesis title, etc.")
			who_measured = st.text_input("*The one who measured or evaluated")
			purity = st.text_input("Material purity")
			pretreatment = st.text_input("Pre-treatment information")
			method_detail = st.text_input("Method in detail")
			description_else = st.text_input("Anything else")
			if not material or not hydrogen or not attribute or not method or not information_source or not who_measured:
				st.warning("It is mandatory to select the options or fill in the information which begin with '*'.")
			else:
				key_col = ['TEMP', hydrogen]
				sub_col = ['TEMP_std', hydrogen + '_std']
				to_suggest, df_suggest = candidate_columns(df, key_col + sub_col)
		else:
			st.empty()
		
	with con11:
		if uploaded_file:
			st.success("Data You Uploaded")
			st.dataframe(df.style.format("{:.2e}"))
			csv = df.to_csv(index=False)
			st.download_button(
				label="Download CSV on your computer",
				data = csv,
				file_name = Path(uploaded_file.name).stem + '.csv',
				mime = 'text/csv'
			)
			if to_suggest == True:
				if all(col in df_suggest for col in key_col):
					st.warning("Your data is converted to be coherent with our database. If you didn't intend this one, modify your own file and upload again.")
					st.success("Converted Data")
					st.dataframe(df_suggest.style.format("{:.3e}"),width=1000)
					if "TEMP" in df_suggest.columns and hydrogen in df_suggest.columns:
						if "TEMP_std" in df_suggest.columns:
							x_err = df_suggest["TEMP_std"]
						else:
							x_err = None
						if hydrogen+"_std" in df_suggest.columns:
							y_err = df_suggest[hydrogen+"_std"]
						else:
							y_err = None
						fig, ax = plt.subplots()
						ax.errorbar(df_suggest["TEMP"], df_suggest[hydrogen], xerr=x_err, yerr=y_err, fmt='o')
						ax.grid(color='gray', ls='-.', lw=0.75)
						ax.set_xlabel("TEMP")
						ax.set_ylabel(hydrogen)
						st.pyplot(fig)
					else:
						st.error(f'Either "TEMP" or "{hydrogen}" not in the converted dataframe')
					save_button = st.button("Save the data on the server.")
				else:
					st.error("Your data must include column names of "+str(key_col))
			elif to_suggest == False:
				if "TEMP" in df_suggest.columns and hydrogen in df_suggest.columns:
					if "TEMP_std" in df_suggest.columns:
						x_err = df_suggest["TEMP_std"]
					else:
						x_err = None
					if hydrogen+"_std" in df_suggest.columns:
						y_err = df_suggest[hydrogen+"_std"]
					else:
						y_err = None
					fig, ax = plt.subplots()
					ax.errorbar(df_suggest["TEMP"], df_suggest[hydrogen], xerr=x_err, yerr=y_err, fmt='o')
					ax.grid(color='gray', ls='-.', lw=0.75)
					ax.set_xlabel("TEMP")
					ax.set_ylabel(hydrogen)
					st.pyplot(fig)
				else:
					st.error(f'Either "TEMP" or "{hydrogen}" not in the dataframe')
				save_button = st.button("Save the data on the server.")

			if save_button == True:
				DWD = make_DWD(material, hydrogen, attribute, method, uploader, df_suggest, information_source, who_measured, purity, pretreatment, method_detail, description_else)
				save_DWD(DWD)
			
		else:
			st.empty()
