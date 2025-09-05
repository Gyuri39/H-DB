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
from utils.session import clear_previous_session, clear_current_session

def createPage():
	st.title("Add data")
	clear_previous_session("AddData")
	if "AddDataStep" not in st.session_state:
		st.session_state.AddDataStep = 1
	else:
		st.empty()

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
	to_suggest = None
	save_button = None
	def nextStep_button(disabled=False):
		nextbutton = st.button("Next step", disabled=disabled)
		if nextbutton:
			st.session_state.AddDataStep += 1
			st.rerun()
	def prevStep_button(things_to_initialize = None):
		prevbutton = st.button("Previous step")
		if prevbutton:
			if things_to_initialize:
				st.session_state[things_to_initialize] = None
			st.session_state.AddDataStep -= 1
			st.rerun()
	def moveStep_buttons(disabled=False):
		col1, col2, col3 = st.columns([1,2,1])
		with col1:
			prevStep_button()
		with col2:
			st.empty()
		with col3:
			nextStep_button(disabled=disabled)
		
	with con11:
		# Step 1: File upload
		if st.session_state.AddDataStep == 1:
			st.subheader("Step 1: File upload")
			st.session_state.AddDataAddedFile = st.file_uploader("Choose a file to upload", type = VALID_DATA_FORMAT, accept_multiple_files = False)
			if st.session_state.AddDataAddedFile:
				st.success("File uploaded")
				disabled=False
			else:
				st.warning("File not uploaded")
				disabled=True
			nextStep_button(disabled)
		# Step 2: Key attributes - selectable
		elif st.session_state.AddDataStep == 2:
			st.subheader("Step 2: Selectable attributes")
			material_default = st.session_state.get("AddDataMaterial", None)
			index = MATERIAL_LIST.index(material_default) if material_default in MATERIAL_LIST else None
			st.session_state.AddDataMaterial = st.selectbox("*Material", MATERIAL_LIST, index=index)
			
			hydrogen_default = st.session_state.get("AddDataHydrogen", None)
			index = list(HYDROGEN_DICT.keys()).index(hydrogen_default) if hydrogen_default in HYDROGEN_DICT.keys() else None
			st.session_state.AddDataHydrogen = st.selectbox("*Form of Hydrogen", HYDROGEN_DICT.keys(), index=index)
			if st.session_state.AddDataHydrogen:
				st.html(HYDROGEN_DICT[st.session_state.AddDataHydrogen])
			
			property_default = st.session_state.get("AddDataAttribute", None)
			index = list(PROPERTY_DICT.keys()).index(property_default) if property_default in PROPERTY_DICT.keys() else None
			st.session_state.AddDataAttribute = st.selectbox("*Property", PROPERTY_DICT.keys(), index=index)
			if st.session_state.AddDataAttribute:
				st.html(PROPERTY_DICT[st.session_state.AddDataAttribute])

			method_default = st.session_state.get("AddDataMethod", None)
			index = METHOD_LIST.index(method_default) if method_default in METHOD_LIST else None
			st.session_state.AddDataMethod = st.selectbox("*Method", METHOD_LIST, index=index)

			datatype_default = st.session_state.get("AddDataDataType", None)
			index = DATA_TYPE_LIST.index(datatype_default) if datatype_default in DATA_TYPE_LIST else None
			st.session_state.AddDataDataType = st.selectbox("*Data type", DATA_TYPE_LIST, index=index)
			if all([st.session_state.AddDataMaterial, st.session_state.AddDataHydrogen, st.session_state.AddDataAttribute, st.session_state.AddDataMethod, st.session_state.AddDataDataType]):
				disabled = False
			else:
				disabled = True
			moveStep_buttons(disabled)
		# Step3: Key attributes - not to be filtered
		elif st.session_state.AddDataStep == 3:
			st.subheader("Step 3: Descriptable attributes")
			st.session_state.AddDataInformationSource = st.text_input("*Information source by DOI, web address, thesis title, etc.", value=st.session_state.get("AddDataInformationSource", None))
			st.session_state.AddDataWhoMeasured = st.text_input("*The one who measured or evalauted", value=st.session_state.get("AddDataWhoMeasured", None))
			st.session_state.AddDataPurity = st.text_input("Material Purity", value=st.session_state.get("AddDataPurity", None))
			st.session_state.AddDataPretreatment = st.text_input("Pre-treatment information", value=st.session_state.get("AddDataPretreatment", None))
			st.session_state.AddDataMethodDetail = st.text_input("Method in detail", value=st.session_state.get("AddDataMethodDetail", None))
			st.session_state.AddDataElseDescription = st.text_input("Anything else", value=st.session_state.get("AddDataElseDescription", None))
			st.session_state.AddDataUploadedPDF = st.file_uploader("Attach a pdf for detailed description", type='pdf', accept_multiple_files = False)
			if all([st.session_state.AddDataInformationSource, st.session_state.AddDataWhoMeasured]):
				if "AddDataSuggested" not in st.session_state:
					st.session_state.AddDataSuggested = None
				if st.session_state.AddDataSuggested:
					disabled = False
				else:
					disabled = True
					st.warning("The input data cannot be properly converted.")
			else:
				disabled = True
				st.warning("It is mandatory to fill in the information which begins with '*'.")
			moveStep_buttons(disabled)
		elif st.session_state.AddDataStep == 4:
			if "AddDataFileSaveTried" not in st.session_state:
				st.session_state.AddDataFileSaveTried = None
			save_button = st.button("Save the data on the server.")
			if save_button == True:
				DWD = make_DWD(
					st.session_state.AddDataMaterial,
					st.session_state.AddDataHydrogen,
					st.session_state.AddDataAttribute,
					st.session_state.AddDataMethod,
					st.session_state.name,
					st.session_state.AddDataSuggestedDF,
					st.session_state.AddDataDataType,
					st.session_state.AddDataInformationSource,
					st.session_state.AddDataWhoMeasured,
					st.session_state.AddDataPurity,
					st.session_state.AddDataPretreatment,
					st.session_state.AddDataMethodDetail,
					st.session_state.AddDataElseDescription
					)
				if st.session_state.AddDataUploadedPDF:
					modify_DWD(DWD, 'pdf_flag', True)
					with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
						tmp.write(st.session_state.AddDataUploaderPDF.read())
						upload_pdf(tmp.name, DWD.date)
				save_DWD(DWD)
				st.session_state.AddDataFileSaveTried = True
			if st.session_state.AddDataFileSaveTried:
				UploadAnotherButton = st.button("Upload another data")
				if UploadAnotherButton:
					st.session_state.AddDataStep = 1
					clear_current_session("AddData",["AddDataStep"])
					st.rerun()
			prevStep_button(AddDataFileSaveTried)

			
	with con12:
		# Step 1-2: File upload and Key attributes select
		if st.session_state.AddDataStep <= 2:
			if st.session_state.AddDataAddedFile:
				if "AddDataAddedDF" not in st.session_state:
					st.session_state.AddDataAddedDF = convert_to_dataframe(st.session_state.AddDataAddedFile)
				st.success("Uploaded Data")
				st.dataframe(st.session_state.AddDataAddedDF.style.format("{:.2e}"))
			else:
				st.session_state.ADddDataAddedDF = None
		# Step 3: Key attributes selected
		elif st.session_state.AddDataStep >= 3:
			key_col = ['TEMP', st.session_state.AddDataHydrogen]
			sub_col = ['TEMP_std', st.session_state.AddDataHydrogen + '_std']
			st.session_state.AddDataSuggested, st.session_state.AddDataSuggestedDF = candidate_columns(st.session_state.AddDataAddedDF, key_col + sub_col)
			df_suggest = st.session_state.AddDataSuggestedDF

			if all(col in df_suggest for col in key_col):
				st.warning("Your data is automatically converted to be coherent with our database. If you didn't intend this one, modify your own file and upload again.")
				st.success("Converted Data")
				st.dataframe(df_suggest.style.format("{:3e}"), width=1000)
				if "TEMP" in df_suggest.columns and st.session_state.AddDataHydrogen:
					if "TEMP_std" in df_suggest.columns:
						x_err = df_suggest["TEMP_std"]
					else:
						x_err = None
					if st.session_state.AddDataHydrogen+"_std" in df_suggest.columns:
						y_err = df_suggest[st.session_state.AddDataHydrogen+"_std"]
					else:
						y_err = None
					fig, ax = plt.subplots()
					ax.errorbar(df_suggest["TEMP"], df_suggest[st.session_state.AddDataHydrogen], xerr=x_err, yerr=y_err, fmt='o')
					ax.grid(color='gray', ls='-.', lw=0.75)
					ax.set_xlabel("TEMP")
					ax.set_ylabel(st.session_state.AddDataHydrogen)
					st.pyplot(fig)
				else:
					st.error(f'Either "TEMP" or "{st.session_state.AddDataHydrogen}" not in the converted dataframe')

			else:
				st.error("Your data must include column names of "+str(key_col))
