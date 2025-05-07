import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from utils.graph_scale import ReciprocalScale
import os
from pathlib import Path
from utils import DATA_DIR, PROPERTY_DICT, VALID_DATA_FORMAT, filter_filelist, save_csv_button, convert_to_dataframe, create_excel_file
from server.data.firestore_handler import load_DWD, info_DWD
from server.data.backblaze_handler import generate_presigned_url
from io import BytesIO
import utils.discussion as discussion
from utils.session import clear_previous_session

def createPage():
	st.title("View Data")
	clear_previous_session("ViewData")

	if "SelectedFileResult" not in st.session_state:
		st.session_state.PreselectedFilterResult = None
	if "__GRAPH_CREATED__" not in st.session_state:
		st.session_state.__GRAPH_CREATED__ = None
	if "DataFrameUploaded" not in st.session_state:
		st.session_state.DataFrameUploaded = None
	if "DataLabels" not in st.session_state:
		st.session_state.DataLabels = {}

	scale_options = ['Linear', 'Logarithmic', 'Reciprocal']
	register_scale(ReciprocalScale)

	con11, con12 = st.columns([0.5, 0.5])
	con21, con22 = st.columns([0.5, 0.5])
	con31, con32 = st.columns([0.5, 0.5])
	con4 = st.columns([1.0])
	con5 = st.columns([1.0])
	
	with con11:
		st.subheader("Filter data files on the server")
		filter_result = filter_filelist()

	with con12:
		st.subheader("Select data to view")
		filtered_container = st.container()
		filtered_list = [name for name in filter_result.filelist]
		select_all = st.checkbox("Select all")
		if select_all:
			selected_options = filtered_container.multiselect("Select one or more data", filtered_list, filtered_list)
		else:
			selected_options = filtered_container.multiselect("Select one or more data", filtered_list)

	with con21:
		if selected_options:
			st.subheader("View data info in detail")
			filter_result.updateFilelist(selected_options)
			st.session_state.PreselectedFilterResult = filter_result
			last_selected = st.radio("Select a file to view info", selected_options)
			if last_selected:
				dwd_object = load_DWD(last_selected)
				with st.expander("See dataframe"):
					st.dataframe(info_DWD(dwd_object, 'data').style.format("{:.3e}"), width=1000)
				con211, con212, con213 = st.columns([3,2,8])
				with con211:
					save_csv_button(info_DWD(dwd_object, 'data'), use_container_width=False)
				with con212:
					st.empty()
				with con213:
					if dwd_object.pdf_flag == True:
						if st.button("Download the attached pdf file for detailed explanation", use_container_width=True):
							pdf_url = generate_presigned_url(dwd_object.date + '.pdf')
							js = f'<script> window.open("{pdf_url}", "_blank"); </script>'
							st.components.v1.html(js)
					else:
						st.empty()
				with st.popover("Open the discussion forum for the selected data", use_container_width=True):
					discussion.commentlist(dwd_object.date)
		else:
			st.empty()

	with con22:
		if selected_options and last_selected:
			dwd_object = load_DWD(last_selected)
			st.success("Detailed information of selected data")
			#st.markdown(f"**{info_DWD(dwd_object, 'material')}, {info_DWD(dwd_object, 'hydrogen')}, {info_DWD(dwd_object, 'attribute')}, {info_DWD(dwd_object, 'method')}, {info_DWD(dwd_object, 'data_type')}**")
			st.markdown(f"Material: {info_DWD(dwd_object, 'material')}")
			st.markdown(f"Hydrogen: {info_DWD(dwd_object, 'hydrogen')}")
			st.html(f"Property: {info_DWD(dwd_object, 'attribute')} - {PROPERTY_DICT[info_DWD(dwd_object, 'attribute')]}")
			st.markdown(f"Method: {info_DWD(dwd_object, 'method')}")
			st.markdown(f"Data type: {info_DWD(dwd_object, 'data_type')}")
			st.markdown(f"Uploaded by *{info_DWD(dwd_object, 'uploader')}* in {info_DWD(dwd_object, 'date')}")
			st.markdown(f"Measured or Evaluated by {info_DWD(dwd_object, 'who_measured')}")
			st.markdown(f"Information source: {info_DWD(dwd_object, 'information_source')}")
			st.markdown(f"Purity information (if any): {info_DWD(dwd_object, 'descript_purity')}")
			st.markdown(f"Pre-treatment information (if any): {info_DWD(dwd_object, 'descript_pretreatment')}")
			st.markdown(f"Method in detail (if any): {info_DWD(dwd_object, 'descript_methoddetail')}")
			st.markdown(f"Else (if any): {info_DWD(dwd_object, 'descript_else')}")
			st.markdown(f"**This data has received {len(info_DWD(dwd_object, 'who_verified'))} verification(s).**")
			st.session_state.DataLabels[last_selected] = st.text_input("\>\> Label in figure \<\<", value = st.session_state.DataLabels[last_selected] if last_selected in st.session_state.DataLabels else last_selected)
			
		else:
			st.empty()
	
	with con31:
		st.subheader("Graph Options")
		upload_compare = st.checkbox("Check this box if you want to plot data **on your computer** together")
		if upload_compare:
			uploaded_file = st.file_uploader("Choose a file to upload", type = VALID_DATA_FORMAT, accept_multiple_files = False)
			if uploaded_file:
				df_upload = convert_to_dataframe(uploaded_file)
				st.session_state.DataFrameUploaded = df_upload
			else:
				st.session_state.DataFrameUploaded = None
		else:
			st.session_state.DataFrameUploaded = None
			st.empty()
	
	with con32:
		if st.session_state.DataFrameUploaded is not None:
			st.dataframe(st.session_state.DataFrameUploaded.style.format("{:.2e}"))
		else:
			st.empty()
			

	with con4[0]:
		if selected_options:
			dataframes = {}
			all_columns = set()
			for file_name in selected_options:
				dataframes[file_name] = info_DWD(load_DWD(file_name), 'data')
				all_columns.update(dataframes[file_name].columns)
			all_columns = list(all_columns)	
			if dataframes:
				tmpXindex = all_columns.index("TEMP") if "TEMP" in all_columns else 0
				tmpYindex = (
					"H" if "H" in all_columns
					else "D" if "D" in all_columns
					else "T" if "T" in all_columns
					else ""
				)
				X_axis = st.selectbox("Select X-axis", all_columns, index=tmpXindex)
				#Y_axis = st.selectbox("Select Y-axis", all_columns, index=tmpYindex)
				Y_axes = st.multiselect("Select Y-axis", all_columns, default=tmpYindex)
				graph_button = st.button("Create Graph")
				if graph_button:
					st.session_state.__GRAPH_CREATED__ = True
				if st.session_state.__GRAPH_CREATED__:
					image_width = st.sidebar.slider("image width", min_value=300, max_value=800, value=500)
					fig, ax = plt.subplots()
					for file_name, df in dataframes.items():
						for Y_axis in Y_axes:
							if X_axis in df.columns and Y_axis in df.columns:
								ax.scatter(df[X_axis], df[Y_axis], label = st.session_state.DataLabels[file_name] if file_name in st.session_state.DataLabels else file_name)
					if st.session_state.DataFrameUploaded is not None:
						df_upload = st.session_state.DataFrameUploaded
						X_upload = st.selectbox("Select X-axis of uploaded file", df_upload.columns, index=0)
						Y_upload = st.selectbox("Select Y-axis of uploaded file", df_upload.columns, index=1)
						ax.scatter(df_upload[X_upload], df_upload[Y_upload], label="Your File")
#					ax.secondary_yaxis('right')
					ax.legend(loc='lower right', bbox_to_anchor=(1.0,1.1))
					
					st.write("Adjust Axis Scaling")
					x_scale = st.selectbox("X-axis scale", scale_options, index=0, key='x_scale')
					y_scale = st.selectbox("Y-axis scale", scale_options, index=0, key='y_scale')
					ax.set_xlabel(X_axis)
					if len(Y_axes) > 1:
						ax.set_ylabel("Multiple isotopes")
					elif len(Y_axes) == 1:
						ax.set_ylabel(Y_axes[0])
					else:
						st.warning("Y axis not selected")
					if x_scale == "Linear":
						ax.set_xscale("linear")
					elif x_scale == "Logarithmic":
						ax.set_xscale("log")
#						ax.secondary_xaxis('top')
					elif x_scale == "Reciprocal":
						ax.set_xscale("reciprocal")
						(xmin, xmax)  = ax.get_xlim()
						ax.set_xlim(xmax, xmin)
						primary_xticks = ax.get_xticks()
						offset_x = 10 ** (np.floor(np.log10(np.max(primary_xticks))) - 1)
#						secondary_xticks = np.arange(np.floor(np.min(primary_xticks)/offset_x) * offset_x, np.ceil(np.max(primary_xticks)/offset_x) * offset_x, offset_x)
#						st.success(primary_xticks)
#						st.success(secondary_xticks)
#						secax = ax.secondary_xaxis('top', (lambda x : x, lambda x: x))
#						secax.xaxis.set_major_locator(AutoLocator())
#						secax.set_xticks(secondary_xticks)
#						secax.set_xticks([1000])
#						secax.set_xlabel(X_axis)
#						secax.xaxis.set_major_locator(MaxNLocator(nbins=len(secondary_xticks)/2))
#						secax.xaxis.set_minor_locator(AutoMinorLocator())
#						secax.set_xticks([1000, 1100])
#						secax.set_xticklabels(secondary_xticks)
#						secax.set_xticklabels([f"{offset_x/t:.2g}" for t in secondary_xticks])
					else:
						raise NotImplementedError(f"x_scale option {x_scale} is not implemented.")
					if y_scale == "Linear":
						ax.set_yscale("linear")
					elif y_scale == "Logarithmic":
						ax.set_yscale("log")
					elif y_scale == "Reciprocal":
						ax.set_yscale("reciprocal")
						(ymin, ymax) = ax.get_ylim()
						ax.set_ylim(ymax, ymin)
					else:
						raise NotImplementedError(f"y_scale option {y_scale} is not implemented.")
					ax.grid(color='gray', ls='-.', lw=0.75)
					buf = BytesIO()
					default_xlim = ax.get_xlim()
					default_ylim = ax.get_ylim()
					with st.expander("Axis range manual setting"):
						x_min_display = st.number_input("X-axis leftmost", value=default_xlim[0], format="%.2e")
						x_max_display = st.number_input("X-axis rightmost", value=default_xlim[1], format="%.2e")
						y_min_display = st.number_input("Y-axis lowermost", value=default_ylim[0], format="%.2e")
						y_max_display = st.number_input("Y-axis uppermost", value=default_ylim[1], format="%.2e")
						ax.set_xlim(x_min_display, x_max_display)
						ax.set_ylim(y_min_display, y_max_display)
					fig.savefig(buf, format="png", bbox_inches="tight")
					buf.seek(0)
					st.image(buf, width=image_width)
					#st.pyplot(fig)
					if st.button("Hide graph"):
						st.session_state.__GRAPH_CREATED__ = False
						st.rerun()

					fig_download = st.download_button(
						label = "Download figure",
						data=buf,
						file_name="figure.png",
						mime="image/png"
					)	

				else:
					st.empty()

		with con5[0]:
			st.subheader("Download selected data")
			if st.checkbox("Download selected data as a single excel file"):
				dfs = []
				for file_name in selected_options:
					st.session_state.DataLabels.setdefault(file_name, file_name)
					dfs.append({"df": info_DWD(load_DWD(file_name), 'data'), "name1": file_name, "name2": st.session_state.DataLabels[file_name]})
				create_excel_file(dfs)
			else:
				st.empty()
