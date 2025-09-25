from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from .base_model import BaseRegressionModel
import streamlit as st
import numpy as np
from io import BytesIO
import joblib
import base64

class LinearRegressionModel(BaseRegressionModel):
	def __init__(self, name=None, degree=1):
		super().__init__(name="Linear", description = "Linear regression with polynomial features")
		self.class_name = self.__class__.__name__
		self.degree = degree
		self.poly = PolynomialFeatures(self.degree, include_bias=False)
		self.model = LinearRegression()
	
	def set_degree(self, degree):
		self.degree = degree
		self.poly = PolynomialFeatures(self.degree, include_bias=False)

	def train(self):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before training.")
		elif self.train_X is None or self.train_y is None:
			raise NotImplementedError("A training set must be initialized before training.")
		elif self.scaler is None:
			raise NotImplementedError("A scaler must be initialized before training.")
		X_train_poly = self.poly.fit_transform(self.scaler.transform(self.apply_transformation(self.train_X, self.feature_transform)))
		y_train = self.apply_transformation(self.train_y, self.target_transform)
		self.model.fit(X_train_poly, y_train, sample_weight=self.sample_weight)
		self.train_score = self.model.score(X_train_poly, y_train)

		feature_names = [f"x{i+1}" for i in range(self.scaler.n_features_in_)]
		poly_feature_names = self.poly.get_feature_names_out(feature_names)
		coefficients = self.model.coef_
		intercept = self.model.intercept_
		terms = [f"{coefficients[0][i]:.3E} * {poly_feature_names[i]}" for i in range(len(coefficients[0]))]
		#terms = [f"{coefficients[i]:.3E} * {poly_feature_names[i]}" for i in range(len(coefficients))]
		equation = " + ".join(terms)
		equation = f"{intercept[0]:.3E} + {equation}"
		#equation = f"{intercept:.3E} + {equation}"
		self.best_params = equation
		st.warning(self.best_params)

	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initilalized before predicting.")

		st.warning("====DEBUG START====")
		st.write(X)
		st.write(self.apply_transformation(X, self.feature_transform))
		st.write(self.scaler.transform(self.apply_transformation(X,self.feature_transform)))
		st.warning("====DEBUG END====")
		X_scaled = self.scaler.transform(self.apply_transformation(X, self.feature_transform))
		X_test_poly = self.poly.transform(X_scaled)
		y_pred = self.model.predict(X_test_poly)
		return self.inverse_transformation(y_pred, self.target_transform)

	def serialize_poly(self):
		poly_buffer = BytesIO()
		joblib.dump(self.poly, poly_buffer)
		poly_bytes = poly_buffer.getvalue()
		return base64.b64encode(poly_bytes).decode('utf-8')

	def to_dict(self):
		base_dict = super().to_dict()
		return {
			**base_dict,
			'poly': self.serialize_poly(),
			'degree': self.degree
		}

	@classmethod
	def deserialize_poly(cls, poly_base64):
		if poly_base64:
			poly_bytes = base64.b64decode(poly_base64.encode('utf-8'))
			poly_buffer = BytesIO(poly_bytes)
			return joblib.load(poly_buffer)
		return None
	
	@classmethod
	def dictload(cls, document_id, collection_name="models"):
		base_dict = BaseRegressionModel.dictload(document_id, collection_name)
		if base_dict:
			data = base_dict.copy()
			data['poly'] = cls.deserialize_poly(data.get('poly'))
			return data
		else:
			print("Base_dict not detected")
		return None

class ElasticNetModel(LinearRegressionModel):
	#def __init__(self, degree=1, alpha=0.0, r=0.0):
	def __init__(self, name=None, degree=1, param_grid=None):
		super().__init__()
		self.class_name = self.__class__.__name__
		self.param_grid = param_grid or {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.0, 0.2, 0.5, 0.8, 1.0]}
		self.name="Elastic"
		description="Elastic net with polynomial features"
		self.model = ElasticNet()
		self.alpha = None
		self.l1_ratio = None

	def set_alpha(self, alpha):
		self.model.alpha = alpha
		self.alpha = alpha

	def set_regularization(self, l1_ratio):
		self.model.l1_ratio = l1_ratio
		self.l1_ratio = l1_ratio

	def train(self):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before training.")
		elif self.train_X is None or self.train_y is None:
			raise NotImplementedError("A training set must be initialized before training.")
		elif self.scaler is None:
			raise NotImplementedError("A scaler must be initialized before training.")
		X_train_poly = self.poly.fit_transform(self.scaler.transform(self.apply_transformation(self.train_X,self.feature_transform)))
		y_train = self.apply_transformation(self.train_y, self.target_transform)
#		if self._yes_gridcv == True or len(self.param_grid['alpha']) >1 or len(self.param_grid['l1_ratio']) >1:
		if self._yes_gridcv == True:
			grid_search = self.grid_search
			grid_search.fit(X_train_poly, self.train_y)
			self.model = grid_search.best_estimator_
			self.train_score = grid_search.best_score_
			self.alpha, self.l1_ratio = grid_search.best_params_['alpha'], grid_search.best_params_['l1_ratio']
		
		else:
			self.model.fit(X_train_poly, y_train, sample_weight=self.sample_weight)
		feature_names = [f"x{i+1}" for i in range(self.scaler.n_features_in_)]
		poly_feature_names = self.poly.get_feature_names_out(feature_names)
		coefficients = self.model.coef_
		intercept = self.model.intercept_
		terms = [f"{coefficients[i]:.3E} * {poly_feature_names[i]}" for i in range(len(coefficients))]
		equation = " + ".join(terms)
		equation = f"{intercept[0]:.3E} + {equation}"
		self.best_params = equation
		self.best_params += f",    alpha: {self.alpha}, l1_ratio: {self.l1_ratio}"
		st.warning(self.best_params)
