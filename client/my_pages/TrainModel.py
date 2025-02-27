import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from utils import DATA_DIR, MODEL_DIR, filter_filelist
from server.data.db_handler import info_DWD, load_DWD
from server.models import LinearRegressionModel, ElasticNetModel, GaussianProcessRegressionModel

def createPage():
	st.title("Generate model")
	if "PreselectedFilterResult" not in st.session_state:
		st.session_state.PreselectedFilterResult = None
	if "DataFrame" not in st.session_state:
		st.session_state.DataFrame = None
	if "__LOADDATA_FLAG__" not in st.session_state:
		st.session_state.__LOADDATA_FLAG__ = False
	if "Model" not in st.session_state:
		st.session_state.Model = None
	con11 = st.columns([1.0])
	con21, con22 = st.columns([0.5, 0.5])
	model_option = ['Linear regression', 'Elastic net', 'Gaussian process regression']
	scaling_option = ['None', 'Standardization', 'Normalization [Min-Max]']


	with con11[0]:
		if st.session_state.PreselectedFilterResult:
			filter_result_inherited = st.session_state.PreselectedFilterResult
			st.success("The data you selected in the **View Data** page is automatically imported.")
			filter_result = filter_filelist(
										DATA_DIR, 
										filter_result_inherited.material,
										filter_result_inherited.attribute,
										filter_result_inherited.method
										)
			filtered_list_inherited = [file.name for file in filter_result_inherited.filelist]
		else:
			filter_result = filter_filelist(DATA_DIR)
			filtered_list_inherited = None
		if filter_result:
			filtered_container = st.container()
			filtered_list = [file.name for file in filter_result.filelist]
			select_all = st.checkbox("Select all")
			if select_all:
				selected_options = filtered_container.multiselect("Select one or more data", filtered_list, filtered_list)
			else:
				selected_options = filtered_container.multiselect("Select one or more data", filtered_list, filtered_list_inherited)

			concat_button = st.button("Data choice completed")
		if concat_button:
			dataframes = {}
			for file_name in selected_options:
				dataframes[file_name] = info_DWD(load_DWD(Path(DATA_DIR) / file_name), 'data')
			common_columns = set(next(iter(dataframes.values())).columns)
			for df in dataframes.values():
				common_columns &= set(df.columns)
			common_columns = list(common_columns)
			dataframes_common = {file_name: df[common_columns] for file_name, df in dataframes.items()}
			df_concat = pd.concat(dataframes_common.values(), ignore_index=True)
			st.session_state.DataFrame = df_concat
			st.session_state.__LOADDATA_FLAG__ = True
		else:
			st.empty()

	if st.session_state.__LOADDATA_FLAG__:
		with con21:
			df = st.session_state.DataFrame
			numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
			if len(numeric_columns) < 2:
				st.error("The dataframe does not contain two or more numeric columns.")
			else:
				features = st.multiselect("Choose features", numeric_columns, numeric_columns[0])
				target = st.selectbox("Choose a target", numeric_columns, index=1)

				if len(features) < 1:
					st.error("Select one or more features.")
				elif not target:
					st.error("Select the target.")
				elif target in features:
					st.error("The target is already incorporated in the features.")
				else:
					X_train = df[features]
					y_train = df[[target]]
					st.success("Training data prepared.")

					model_chosen = st.selectbox('Select regression model', model_option, index=2)
					scaling_chosen = st.selectbox('Select data scaling option', scaling_option, index=2)
					#Linear regression model
					if model_chosen == model_option[0]:
						model = LinearRegressionModel()
						st.write(model.description)
						model.set_data(X_train, y_train)
						model.set_scaler(scaling_chosen)
						degree = st.number_input('polynomial degree of the model:', value=1, format="%d")
						model.set_degree(degree)
					#Elastic net model
					elif model_chosen == model_option[1]:
						model = ElasticNetModel()
						st.write(model.description)
						model.set_data(X_train, y_train)
						model.set_scaler(scaling_chosen)
						degree = st.number_input('polynomial degree of the model:', value=1, format="%d")
						alpha = st.number_input('Penalty term $alpha$', value=1e-5)
						r =st.slider('Regularization factor $r$', 0.0, 1.0, value=0.5)
						model.set_degree(degree)
						model.set_alpha(alpha)
						model.set_r(r)
					#Gaussian process regression model
					elif model_chosen == model_option[2]:
						model = GaussianProcessRegressionModel()
						st.write(model.description)
						model.set_data(X_train, y_train)
						model.set_scaler(scaling_chosen)
						kernel_option = st.selectbox('Chosse the kernel option', ['Single', 'Multiple'], index=0)
						whitekernel_option = st.selectbox('Choose the white kernel option', ['Use', 'Not use'], index=0)
						n_restarts = st.number_input('Number of restarts of the optimizer', value=1, format="%d")
						
						if kernel_option == 'Single':
							yes_multiple = False
						elif kernel_option == 'Multiple':
							yes_multiple = True
						else:
							raise KeyError("Invalid kernel_option designation")
						if whitekernel_option == 'Use':
							yes_white = True
						elif whitekernel_option == 'Not use':
							yes_white = False
							noise_std_model = st.number_input('Fixed noise level', value = 1e-5)
							set_alpha(noise_std_model)
						else:
							raise KeyError("Invalid whitekernel_option designation")
						model.set_kernel(yes_multiple, yes_white)
					else:
						raise KeyError("Invalid model designation")
					#Train
					if st.checkbox("Fit with the above setting"):
						model.train()
						st.success("Model is successfully trained.")
						st.session_state.Model = model
						current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
						filename = f"{model.name}_{st.session_state.name}_{current_time}.pickle"
						modeldir = Path(MODEL_DIR) / filename
						model.set_training_info(
							f"""
								Training set: {selected_options}
								Features: {features}
								Target: {target}
							"""
						)
						description_others = st.text_input("Write anything you want to describe the model.")
						if st.button("Save the model"):
							model.set_description_others(description_others)
							st.session_state.Model = model
							model.save(modeldir)
							st.success(f"File '{filename}' is saved.")

		with con22:
			if st.session_state.Model:
				model = st.session_state.Model
				Xoption = st.selectbox('Select one feature', features, index=0)
				fig, ax = plt.subplots()
				ax.scatter(model.train_X[Xoption], model.train_y, marker='s', color='black', alpha=0.5, label='True value', s=20)
				# yes deviations
				if model._yes_dev is True:
					y_train_predicted, ystd_train_predicted = model.predict(model.train_X)
					sorted_indices = np.argsort(model.train_X[Xoption])
					sorted_X = model.train_X[Xoption][sorted_indices]
					sorted_y_predicted = y_train_predicted[sorted_indices]
					sorted_ystd_predicted = ystd_train_predicted[sorted_indices]
					ax.errorbar(sorted_X, sorted_y_predicted, yerr=sorted_ystd_predicted, color='blue', alpha=0.3)
				# no deviations
				elif model._yes_dev is False:
					y_train_predicted = model.predict(model.train_X)
				else:
					raise NotImplementedError("A model deviation is not designated")
				fix, ax = plt.subplots()
				ax.scatter(model.train_X[Xoption], y_train_predicted, marker='o', color='red', alpha=0.5, label='Predicted value', s=20)
				ax.set_xlabel(Xoption)
				ax.set_ylabel(target)
				ax.set_title(f"{Xoption} vs {target}")
				ax.legend()
				st.pyplot(fig)
	else:
		st.empty()
