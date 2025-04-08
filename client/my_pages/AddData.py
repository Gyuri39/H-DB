import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from utils import DATA_DIR, MATERIAL_LIST, PROPERTY_DICT, METHOD_LIST, DATA_TYPE_LIST, HYDROGEN_DICT, VALID_DATA_FORMAT, convert_to_dataframe
from server.data.firestore_handler import make_DWD, save_DWD, modify_DWD
from server.data.backblaze_handler import upload_pdf
from server.data.utils import candidate_columns

def createPage():
	st.title("Add data")
	with st.popover("Upload Guide"):
		st.markdown("### Upload Guide ###")
		st.markdown("*Upload a data with a proper format.")
		st.markdown("**A csv file or excel/txt file that mimics the csv format can be uploaded.")
		st.image("files/fig/add_data_fig_excel.png", caption="Upload format example")
		st.markdown("(a) Property of **hydrogen (H)** along with **temperature (Temp)**")
		st.markdown("(b) Error in **standard deviation** for both Temp **(Temp_std)** and H **(H_std)** in addition to (a)")
		st.markdown("(c) Several columns in the raw data; will be **converted** to the proper format when uploading")
		st.markdown("*Please carefully confirm before uploading the file if the conversion has been done as expected.")
		with st.expander("See converted data"):
			st.markdown("(a) and (b) are the perfectly fit format with the database")
			st.image("files/fig/add_data_fig_a_converted.png", caption="(a) converted to as it is")
			st.image("files/fig/add_data_fig_b_converted.png", caption="(b) converted to as it is")
			st.markdown("(c) is not perfectly fit, but can be converted to a proper format")
			st.image("files/fig/add_data_fig_c_converted.png", caption="H selected as the form of hydrogen")
		st.markdown("*After uploading the file, choose/write appropriate attributes of that data.")
		st.image("files/fig/add_data_fig_attributes.png", caption="Parts of mandatory input")
		st.markdown("**Choose the Material; if there is not, please contact us.")
		st.markdown("**Choose the Form of Hydrogen, either in isotope(atom) of isotopologue(molecule).")
		st.markdown("**Choose the Property. Be careful of the unit.")
		with st.expander("Unit of each property"):
			for prop in PROPERTY_DICT:
				st.html(f"{prop} : {PROPERTY_DICT[prop]}")
		st.markdown("**Choose the Method the data was obtained, by experiment, simulation, or theory/model")
		st.markdown("**Choose the Data type in which the data is obtained, either as the raw data itself or from fitted curve.")

	con11, con12 = st.columns([0.5, 0.5])
	uploaded_file = st.file_uploader("Choose a file to upload", type = VALID_DATA_FORMAT, accept_multiple_files = False)
	to_suggest = None
	save_button = None
	if uploaded_file:
		df = convert_to_dataframe(uploaded_file)
	with con12:
		if uploaded_file:
			material = st.selectbox("*Material", MATERIAL_LIST, index=None)
			hydrogen = st.selectbox("*Form of Hydrogen", HYDROGEN_DICT.keys(), index=None)
			if hydrogen:
				st.html(HYDROGEN_DICT[hydrogen])
			attribute = st.selectbox("*Property", PROPERTY_DICT.keys(), index=None)
			if attribute:
				st.html(PROPERTY_DICT[attribute])
			method = st.selectbox("*Method", METHOD_LIST, index=None)
			data_type = st.selectbox("*Data type", DATA_TYPE_LIST, index=None)
			uploader = st.session_state.name
			information_source = st.text_input("*Information source by DOI, web address, thesis title, etc.")
			who_measured = st.text_input("*The one who measured or evaluated")
			purity = st.text_input("Material purity")
			pretreatment = st.text_input("Pre-treatment information")
			method_detail = st.text_input("Method in detail")
			description_else = st.text_input("Anything else")
			uploaded_pdf = st.file_uploader("Attach a pdf for detailed description", type = 'pdf', accept_multiple_files = False)
			if not material or not hydrogen or not attribute or not method or not data_type or not information_source or not who_measured:
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
				DWD = make_DWD(material, hydrogen, attribute, method, uploader, df_suggest, data_type, information_source, who_measured, purity, pretreatment, method_detail, description_else)
				if uploaded_pdf:
					modify_DWD(DWD, 'pdf_flag', True)
					with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
						tmp.write(uploaded_pdf.read())
						upload_pdf(tmp.name, DWD.date)
				save_DWD(DWD)
			
		else:
			st.empty()
