import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from utils.graph_scale import ReciprocalScale
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from utils import DATA_DIR, MODEL_DIR, filter_filelist
from server.data.db_handler import info_DWD, load_DWD
from server.models import LinearRegressionModel, ElasticNetModel, GaussianProcessRegressionModel
from io import BytesIO

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

	scale_options = ['Linear', 'Logarithmic', 'Reciprocal']
	register_scale(ReciprocalScale)


	with con11[0]:
		if st.session_state.PreselectedFilterResult:
			filter_result_inherited = st.session_state.PreselectedFilterResult
			st.success("The data you selected in the **View Data** page is automatically imported.")
			filter_result = filter_filelist(
										material = filter_result_inherited.material,
										hydrogen = filter_result_inherited.hydrogen,
										attribute = filter_result_inherited.attribute,
										method = filter_result_inherited.method
										)
			filtered_list_inherited = filter_result_inherited.filelist
		else:
			filter_result = filter_filelist()
			filtered_list_inherited = None
		if filter_result:
			filtered_container = st.container()
			filtered_list = filter_result.filelist
			select_all = st.checkbox("Select all")
			if select_all:
				selected_options = filtered_container.multiselect("Select one or more data", filtered_list, filtered_list)
			else:
				selected_options = filtered_container.multiselect("Select one or more data", filtered_list, filtered_list_inherited)

			concat_button = st.button("Data choice completed")
		if concat_button:
			dataframes = {}
			for file_name in selected_options:
				dataframes[file_name] = info_DWD(load_DWD(file_name), 'data')
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
				features = st.multiselect("Choose one or more features", numeric_columns, "TEMP" if "TEMP" in numeric_columns else None)
				target = st.selectbox("Choose a target", numeric_columns, index=0)

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
						alpha = st.number_input('Penalty term $alpha$', value=1e-10, format="%.3e")
						regularization = st.slider('Regularization factor $r$', 0.0, 1.0, value=0.5)
						model.set_degree(degree)
						model.set_alpha(alpha)
						model.set_regularization(regularization)
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
							noise_std_model = st.number_input('Fix the noise level', value=1e-10, format="%e")
							model.set_alpha(noise_std_model)
						elif kernel_option == 'Multiple':
							yes_multiple = True
						else:
							raise KeyError("Invalid kernel_option designation")
						if whitekernel_option == 'Use':
							yes_white = True
							noise_min, noise_max = st.slider(
								"Set white noise level bounds",
								min_value=-20, max_value=20, value=(-10, 5), step=1, format="10^%d"
							)
							model.set_noise_bounds(10**noise_min, 10**noise_max)
						elif whitekernel_option == 'Not use':
							yes_white = False
						else:
							raise KeyError("Invalid whitekernel_option designation")
						model.set_RBFkernel(yes_multiple, yes_white)
					else:
						raise KeyError("Invalid model designation")
					#Train
					if st.checkbox("Fit with the above setting"):
						model.train()
						st.success("Model is successfully trained.")
						st.session_state.Model = model
						current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
						filename = f"{model.name}_{st.session_state.name}_{current_time}.model"
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
							model.save(filename)
							st.success(f"File '{filename}' is saved.")
					else:
						st.session_state.Model = None

		with con22:
			if st.session_state.Model:
				model = st.session_state.Model
				Xoption = st.selectbox('Select one feature', features, index=0)
				x_scale = st.selectbox("X-axis scale", scale_options, index=0, key='x_scale')
				y_scale = st.selectbox("Y-axis scale", scale_options, index=0, key='y_scale')
				
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
				
				ax.scatter(model.train_X[Xoption], y_train_predicted, marker='o', color='red', alpha=0.5, label='Predicted value', s=20)
				ax.set_title(f"{Xoption} vs {target}")
				ax.set_xlabel(Xoption)
				ax.set_ylabel(target)
				
				# Axis Scaling
				if x_scale == "Linear":
					ax.set_xscale("linear")
				elif x_scale == "Logarithmic":
					ax.set_xscale("log")
				elif x_scale == "Reciprocal":
					ax.set_xscale("reciprocal")
					(xmin, xmax) = ax.get_xlim()
					ax.set_xlim(xmax, xmin)
				else:
					raise NotImplementedError(f"x_scale option {x_scale} is not implemented.")

				if y_scale == "Linear":
					ax.set_yscale("linear")
				elif y_scale == "Logarithmic":
					ax.set_yscale("log")
				elif y_scale == "Reciprocal":
					ax.set_yscale("reciprocal")
					(ymin, ymax) = ax.get_ylim()
					ax.set_xlim(ymax, ymin)
				else:
					raise NotImplementedError(f"y_scale option {y_scale} is not implemented.")

				ax.grid(color='gray', ls='-.', lw=0.75)
				ax.legend()
				buf = BytesIO()
				fig.savefig(buf, format="png", bbox_inches="tight")
				buf.seek(0)
				st.image(buf)
				#st.pyplot(fig)
	else:
		st.empty()
