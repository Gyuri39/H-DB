import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from utils.graph_scale import ReciprocalScale
from datetime import datetime
from io import BytesIO
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils import filter_filelist
from server.data.firestore_handler import info_DWD, load_DWD
from server.models import GaussianProcessRegressionModel
from utils.session import clear_previous_session

def createPage():
	st.title("Train with GPy (and convert to scikit-learn)")
	clear_previous_session("TrainGPR")
	register_scale(ReciprocalScale)

	filter_result = filter_filelist()
	if not filter_result:
		st.error("No data available.")
		return
	filtered_list = filter_result.filelist
	select_all = st.checkbox("Select all")
	default_sel = filtered_list if select_all else []
	selected_files = st.container().multiselect("Select one or more data", filtered_list, default_sel, format=lambda fn: st.session_state.get(fn, fn))
	if not selected_files:
		return

	concat_checkbox = st.checkbox("Load and concatenate selected data")
	if not concat_checkbox:
		return

	# Load data
	dataframes = [info_DWD(load_DWD(fp), 'data') for fp in selected_files]
	common_columns = set.intersection(*(set(df.columns) for df in dataframes))
	common_columns = sorted(common_columns)

	if len(common_columns) < 2:
		st.error("Not enough common numeric columns")
		return

	df_concat = pd.concat([df[common_columns] for df in dataframes], ignore_index=True)
	numeric_cols = [col for col in df_concat.columns if pd.api.types.is_numeric_dtype(df_concat[col])]

	features = st.multiselect("Select feature(s)", numeric_cols, "TEMP" if "TEMP" in numeric_cols else None)
	target = st.selectbox("Select target", numeric_cols)

	if not features or target in features:
		st.error("Invalid feature/target selection")
		return

	# Data preprocessing
	transform_options = ["linear", "log", "reciprocal"]
	feature_transform = {}
	with st.expander("Data transform option"):
		for i, feature in enumerate(features):
			feature_transform[feature] = st.selectbox(f"Feature **{feature}**", transform_options, index=0, key=f"{feature}_transform")
		target_transform = st.selectbox(f"Target **{target}**", transform_options, index=0, key="target_transform")

	def apply_transform(x, method):
		if method == "linear":
			return x
		elif method == "log":
			return np.log(np.clip(x, 1e-30, None))
		elif method == "reciprocal":
			return 1.0 / np.clip(x, 1e-30, None)
		else:
			raise ValueError("Invalid transform option")
	def reverse_transform(x, method):
		if method == "linear":
			return x
		elif method == "log":
			return np.exp(x)
		elif method == "reciprocal":
			return 1.0 / x
		else:
			raise ValueError("Invliad reverse transform option")
	
	Xorig = df_concat[features].copy()
	yorig = df_concat[target].copy()
	X = df_concat[features].copy()
	for col in features:
		X[col] = apply_transform(X[col].values, feature_transform[col])
	X = X.to_numpy()
	y = df_concat[target].copy()
	y = apply_transform(y.values, target_transform).reshape(-1, 1)
	label_map = {}
	for idx, name in enumerate(selected_files):
		n = len(info_DWD(load_DWD(name), 'data'))
		label_map[name] = np.full(n, idx)
	dataset_label = np.concatenate(list(label_map.values()))

	# GPy model
	gpy_approach = st.selectbox("Unknown uncertainty", ["by data point", "by dataset"], index=0)

	if gpy_approach == "by dataset":
		num_groups = len(set(dataset_label))
		d = X.shape[1]
		rbf = GPy.kern.RBF(input_dim=d, variance=1.0, lengthscale=1.0, active_dims=list(range(d)))
		coreg = GPy.kern.Coregionalize(input_dim=1, output_dim=num_groups, rank=num_groups, active_dims=[d])
		white = GPy.kern.White(input_dim=1, active_dims=[d])
		white_coreg = white.prod(coreg)
		kern = rbf.prod(coreg) + white_coreg
		X_aug = np.hstack([X, dataset_label.reshape(-1,1)])
		model_gpy = GPy.models.GPRegression(X_aug, y, kernel=kern)

	elif gpy_approach == "by data point":
		kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1.0, lengthscale=1.0)
		init_sigma = np.ones(len(selected_files)) * 0.001

		def make_noise_vector(sigma_vals, labels):
			noise = np.zeros(len(labels))
			for k in range(len(sigma_vals)):
				noise[labels == k] = sigma_vals[k] ** 2
			return noise.reshape(-1, 1)

		noise_vector = make_noise_vector(init_sigma, dataset_label)
		model_gpy = GPy.models.GPHeteroscedasticRegression(X, y, kernel=kernel)
		noise_param = [k for k in model_gpy.parameter_names() if 'Gaussian_noise.variance' in k]
		if not noise_param:
			noise_param = [k for k in model_gpy.parameter_names() if 'het_Gauss' in k and 'variance' in k]
		if not noise_param:
			raise AttributeError("GPy model cannot find noise variance parameter")
		noise_key = noise_param[0]
		model_gpy[noise_key] = noise_vector

	if st.button("Train GPy model"):
		model_gpy.optimize(messages=True, max_iters=1000)
		if gpy_approach == "by dataset":
			model_gpy.optimize_restarts(num_restarts=100, verbose=True, parallel=True)
		st.success("GPy training complete")

		# Convert to scikit-learn
		if gpy_approach == "by dataset":
			prod_rbf_coreg = model_gpy.kern.parts[0]
			rbf = prod_rbf_coreg.parts[0]
			rbf_lengthscale = rbf.lengthscale.values.flatten()
			rbf_variance = float(rbf.variance.values)

			prod_white_coreg = model_gpy.kern.parts[1]
			white = prod_white_coreg.parts[0]
			coreg = prod_white_coreg.parts[1]

			white_variance = float(white.variance.values)
			W = coreg.W.values
			kappa = coreg.kappa.values

			B = W.dot(W.T) + np.diag(kappa)
			learned_group_noise = white_variance * np.diag(B)
			alpha_per_sample = learned_group_noise[dataset_label.astype(int)]
			skl_kernel = C(rbf_variance, constant_value_bounds="fixed") * RBF(length_scale=rbf_lengthscale, length_scale_bounds="fixed")
			skl_model = GaussianProcessRegressor(kernel=skl_kernel, alpha=alpha_per_sample, optimizer=None, normalize_y=False)

		elif gpy_approach == "by data point":
			learned_noise = model_gpy[noise_key].values.flatten()
			rbf_lengthscale = model_gpy.kern.lengthscale.values.flatten()
			rbf_variance = float(model_gpy.kern.variance.values)
			skl_kernel = C(rbf_variance, constant_value_bounds="fixed") * RBF(rbf_lengthscale, length_scale_bounds="fixed")
			skl_model = GaussianProcessRegressor(kernel=skl_kernel, alpha=learned_noise, optimizer=None, normalize_y=True)
		skl_model.fit(X, y.ravel())
		st.success("Converted to scikit-learn GPR model")
		st.warning(f"RBF kernel: {skl_model.kernel_}")
		st.warning(f"White kernel: {skl_model.alpha}")
		st.warning(f"r2 score: {skl_model.score(X,y)}")
		st.success(f"LML: {skl_model.log_marginal_likelihood()}")

		# Save
		gpr_model = GaussianProcessRegressionModel()
		gpr_model.model = skl_model
		gpr_model.set_data(pd.DataFrame(Xorig, columns=features), pd.Series(yorig.ravel()))
		gpr_model.set_transform([feature_transform[key] for key in features], target_transform)
		gpr_model.set_scaler("None")
		gpr_model._yes_dev = True

		now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		filename = f"GPyGPR_{now}.model"
		if st.button("Save model"):
			gpr_model.save(filename)
			st.success(f"Model saved as {filename}")

		# Plotting
		x_col = features[0]
		fig, ax = plt.subplots()
		x_vals = Xorig.iloc[:, 0]
		y_pred, y_err = gpr_model.predict(Xorig)
		#print(f"ypred: {y_pred}")
		#print(f"yerr: {y_err}")
		#ax.errorbar(x_vals, y_pred.values.flatten(), yerr=y_std.values.flatten(), fmt='o', alpha=0.5, label="Prediction")
		ax.errorbar(
			x_vals, 
			y_pred.values.flatten(), 
			yerr=(
				y_err[0].values.flatten() if hasattr(y_err[0], "values") else y_err[0].flatten(), 
				y_err[1].values.flatten() if hasattr(y_err[1], "values") else y_err[1].flatten()
			), 
			fmt='o', 
			alpha=0.5, 
			label="Prediction"
		)
		ax.scatter(x_vals, yorig, color='black', s=10, label="True")

		if len(features) == 1:
			x_grid = np.linspace(x_vals.min(), x_vals.max(), 200).reshape(-1,1)
			ygrid_mean, ygrid_err = gpr_model.predict(pd.DataFrame(x_grid, columns=[x_col]))
			x_grid = x_grid.flatten()
			ygrid_mean = ygrid_mean.values.flatten()
			ygrid_err = (
				ygrid_err[0].values.flatten() if hasattr(ygrid_err[0], "values") else ygrid_err[0].flatten(),
				ygrid_err[1].values.flatten() if hasattr(ygrid_err[1], "values") else ygrid_err[1].flatten()
			)
			ax.plot(x_grid, ygrid_mean, color='blue', alpha=0.3, label="Mean (grid)")
			ax.fill_between(x_grid, ygrid_mean - ygrid_err[0], ygrid_mean + ygrid_err[1], color='blue', alpha=0.1, label="1 sigma reliability")

		ax.set_xlabel(x_col)
		ax.set_ylabel(target)
		ax.legend()
		x_scale = feature_transform[x_col]
		y_scale = target_transform
		ax.set_xscale(x_scale)
		if x_scale == "reciprocal":
			(xmin, xmax) = ax.get_xlim()
			ax.set_xlim(xmax, xmin)
		ax.set_yscale(y_scale)
		if y_scale == "reciprocal":
			(ymin, ymax) = ax.get_ylim()
			ax.set_ylim(ymax, ymin)
		st.pyplot(fig)

		# DEBUG: comparing two models
		# ───────────────────────────────
		# 1. 그리드 생성 (1-D 입력)
		# ───────────────────────────────
		x_col   = features[0]
		x_vals  = Xorig[x_col]
		RES     = 200                                   # 그리드 해상도
		x_grid  = np.linspace(x_vals.min(),
		                      x_vals.max(),
		                      RES).reshape(-1, 1)        # shape = (RES,1)
		df_grid = pd.DataFrame(x_grid, columns=[x_col])

		# ───────────────────────────────
		# 2. scikit-learn 예측
		# ───────────────────────────────
		skl_mean, skl_err = gpr_model.predict(df_grid)   # (μ, (σ−, σ+))
		skl_mu   = skl_mean.values.flatten()
		skl_lo   = (skl_err[0].values.flatten() if hasattr(skl_err[0], "values")
		            else skl_err[0].flatten())
		skl_up   = (skl_err[1].values.flatten() if hasattr(skl_err[1], "values")
				    else skl_err[1].flatten())

		# ───────────────────────────────
		# 3. GPy 예측
		#    GPy 반환은 (μ, σ²)이 기본이므로 sqrt 변환
		# ───────────────────────────────
		#gpy_mu, gpy_var = model_gpy.predict(x_grid)      # ndarray 반환
		#gpy_mu = gpy_mu.flatten()
		#gpy_sd = np.sqrt(gpy_var.flatten())              # 대칭 표준편차
		#gpy_lo = gpy_mu - gpy_sd                         # 1 σ 구간
		#gpy_up = gpy_mu + gpy_sd
		all_preds = []
		for col in features:
			x_grid_transformed = apply_transform(x_grid, feature_transform[x_col])
		for group_index in range(num_groups):
			x_grid_aug = np.hstack([
				x_grid_transformed,
				np.full((x_grid.shape[0],1), group_index)
			])
			print(f"xgridaug with group index {group_index}")
			print(x_grid_aug)
			gpy_mu, gpy_var = model_gpy.predict(x_grid_aug)
			gpy_mu = gpy_mu.flatten()
			gpy_sd = np.sqrt(gpy_var.flatten())
			gpy_lo = gpy_mu - gpy_sd
			gpy_up = gpy_mu + gpy_sd
			df_pred = pd.DataFrame({
				x_col	: x_grid.flatten(),
				"group"	: group_index,
				"μ_GPy"	: reverse_transform(gpy_mu, target_transform),
				"lo_GPy(−1σ)"  : reverse_transform(gpy_lo, target_transform),
				"up_GPy(+1σ)"  : reverse_transform(gpy_up, target_transform),
				"μ_skl"        : skl_mu,
				"lo_skl(−1σ)"  : skl_lo,
				"up_skl(+1σ)"  : skl_up,
			})
			all_preds.append(df_pred)
		df_all = pd.concat(all_preds, ignore_index=True)

		# ───────────────────────────────
		# 5. 출력
		# ───────────────────────────────
		float_cols = df_pred.select_dtypes(include="float").columns
		for col in float_cols:
			df_all[col] = df_all[col].apply(lambda x: f"{x:.5e}")
		st.write(f"### Grid prediction comparison ({RES} points)")
		st.dataframe(df_all)          # 또는 st.table(df_pred.head())

		fig1, ax1 = plt.subplots()

	else:
		st.empty()
