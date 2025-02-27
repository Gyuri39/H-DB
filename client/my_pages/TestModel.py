import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from utils.graph_scale import ReciprocalScale
import os
import numpy as np
import itertools
from pathlib import Path
from utils import MODEL_DIR, save_csv_button
from server.models import BaseRegressionModel, LinearRegressionModel, ElasticNetModel, GaussianProcessRegressionModel

def createPage():
	st.title("Apply model")
	if "Model" not in st.session_state:
		st.session_state.Model = None
	test_df = False
	scale_options = ['Linear', 'Logarithmic', 'Reciprocal']
	register_scale(ReciprocalScale)
	con11, con12 = st.columns([0.5, 0.5])
	con2 = st.columns([1.0])

	with con11:
		st.subheader("Model selection")
		all_models = list(Path(MODEL_DIR).glob("*.pickle"))
		all_models_filename = [os.path.basename(str(dirname)) for dirname in all_models]
		selected_model_name = st.selectbox("Select a model", all_models_filename)
		if selected_model_name:
			selected_model = BaseRegressionModel.load(Path(MODEL_DIR) / selected_model_name)
			st.session_state.Model = selected_model
			train_X = selected_model.train_X
			train_y = selected_model.train_y
			feature_names = train_X.columns
			target_name = train_y.columns[0]
			st.write(f"feature names {feature_names}")
			st.write(f"target name {target_name}")
			df_min = train_X.min()
			df_max = train_X.max()
			
			want_test = st.selectbox("What do you want to evaluate with the model?", ["A point", "Training set itself", "Test set", "Grid points"])
			if want_test == "A point":
				test_dict = {}
				for name in feature_names:
					test_dict[name] = st.number_input(f"Enter the value for {name}")
				test_df = pd.DataFrame([test_dict])
			elif want_test == "Training set itself":
				test_df = train_X
			elif want_test == "Test set":
				st.error("To be implemented")
				test_df = False
			elif want_test == "Grid points":
				grid_row = [2] * len(train_X.columns)
				st.error("DEBUG")
				df_combined = pd.DataFrame([df_min.values, df_max.values, grid_row], index=['min', 'max', 'number of grid'], columns=train_X.columns)
				test_df_grid = st.data_editor(df_combined)
				if (test_df_grid.loc['number of grid'] < 2).any():
					st.warning("The number of grid points must be larger than or equal to 2.")
				else:
					grid = {}
					for name in feature_names:
						grid[name] = np.linspace(test_df_grid[name]['min'], test_df_grid[name]['max'], int(test_df_grid[name]['number of grid']))
					combinations = itertools.product(*grid.values())
					test_df = pd.DataFrame(combinations, columns=grid.keys())
			else:
				raise ValueError("want_test value is out of scope")

	with con12:
		st.subheader("Selected model detail")
		if selected_model:
			st.write(selected_model.description)
			st.write(selected_model.scaler)
			st.write(selected_model.training_info)
			st.markdown(selected_model.description_others)
	
	with con2[0]:
		if st.checkbox("Evalutate") and test_df is not False:
			st.subheader("Evaluation result")
			if selected_model._yes_dev == True:
				test_y, test_ystd = selected_model.predict(test_df)
				df_predicted = pd.DataFrame({
							"Prediction": test_y,
							"Prediction_std": test_ystd
				})
			elif selected_model._yes_dev == False:
				test_y = selected_model.predict(test_df)
				df_predicted = pd.DataFrame({"Prediction": test_y})
			else:
				raise NotImplementedError("Model is not implemented.")
			df_to_download = pd.concat([test_df, df_predicted], axis=1)
			save_csv_button(df_to_download)

			Xoption = st.selectbox("Select one feature", feature_names)
			fig, ax = plt.subplots()

			ax.scatter(train_X[[Xoption]], train_y, marker='s', color='black', alpha=0.3, label='True value', s=20)
			if selected_model._yes_dev == True:
				sorted_indices = np.argsort(test_df[Xoption])
				sorted_Xtest = test_df.iloc[sorted_indices]
				sorted_y_predicted = test_y[sorted_indices]
				sorted_ystd_predicted = test_ystd[sorted_indices]
				ax.errorbar(sorted_Xtest[Xoption].values, sorted_y_predicted, yerr=sorted_ystd_predicted, color='blue', alpha=0.2)
			ax.scatter(test_df[Xoption], test_y, marker='o', color='red', alpha=0.7, label='Predicted value', s=20)
			ax.secondary_xaxis('top')
			ax.secondary_yaxis('right')
			ax.legend()

			st.write("Adjust Axis Scaling")
			x_scale = st.selectbox("X-axis scale", scale_options, index=0, key='x_scale')
			y_scale = st.selectbox("Y-axis scale", scale_options, index=0, key='y_scale')
			if x_scale == "Linear":
				ax.set_xscale("linear")
				ax.set_xlabel(Xoption)
			elif x_scale == "Logarithmic":
				ax.set_xscale("log")
				ax.set_xlabel(Xoption)
			elif x_scale == "Reciprocal":
				ax.set_xscale("reciprocal")
				ax.set_xlabel(f"1/{Xoption}")
				(xmin, xmax) = ax.get_xlim()
				ax.set_xlim(xmax, xmin)
				ReciprocalScale(ax.xaxis).adjust_offset(ax.xaxis)
			else:
				raise NotImplementedError(f"x_cale option {x_scale} is not implemented.")
			if y_scale == "Linear":
				ax.set_yscale("linear")
				ax.set_ylabel(target_name)
			elif y_scale == "Logarithmic":
				ax.set_yscale("log")
				ax.set_ylabel(target_name)
			elif y_scale == "Reciprocal":
				ax.set_yscale("reciprocal")
				ax.set_ylabel(f"1/{target_name}")
				(ymin, ymax) = ax.get_ylim()
				ax.set_ylim(ymax, ymin)
				ReciprocalScale(ax.yaxis).adjust_offset(ax.yaxis)
			else:
				raise NotImplementedError(f"y_scale option {y_scale} is not implemented.")
			ax.grid(color='gray', ls='-.', lw=0.75, alpha=0.2)
			ax.set_title(f"{Xoption} vs {target_name}")
			ax.legend()
			st.pyplot(fig)
			
		else:
			st.empty()
		
