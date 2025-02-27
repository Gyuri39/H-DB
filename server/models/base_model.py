import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

class BaseRegressionModel:
	def __init__(self, name, description=""):
		self.name = name
		self.description = description
		self.model = None
		self.train_X = None
		self.train_y = None
		self.scaler = None
		self._yes_dev = False
		self.training_info = None
		self.description_others = None
		self._yes_gridcv = False
		self.param_grid = None
		self.grid_search = None
		self.best_params = None
		self.train_score = None
		self.test_score = None
	
	def set_training_info(self, description:str):
		self.training_info = description

	def set_description_others(self, description:str):
		self.description_others = description

	def set_data(self, train_X: pd.DataFrame, train_y: pd.DataFrame):
		self.train_X = train_X
		self.train_y = train_y
	
	def set_scaler(self, scaler_name):
		if self.train_X is None:
			raise NotImplementedError("A training set must be initialized before scaling.")
		if scaler_name.startswith("Standard"):	#Standardize
			self.scaler = StandardScaler().fit(self.train_X)
		elif scaler_name.startswith("Normal"):	#Min-Max Scale or Normalize
			self.scaler = MinMaxScaler().fit(self.train_X)
		elif scaler_name == "None": #No Scaling
			scaler = StandardScaler().fit(self.train_X)
			scaler.mean_ = np.zeros(self.train_X.shape[1])
			scaler.scale_ = np.ones(self.train_X.shape[1])
			self.scaler = scaler
		else:
			raise KeyError("Invalid scaler import trial")

	def set_gridsearch(self, param_grid):
		self._yes_gridcv = True
		self.grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', n_jobs=None)
			
	def train(self):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before training.")
		elif self.train_X is None or self.train_y is None:
			raise NotImplementedError("A training set must be initialized before training.")
		elif self.scaler is None:
			raise NotImplementedError("A scaler must be initialized before training.")
		X_train = self.scaler.transform(self.train_X)
		if self._yes_gridcv:
			grid_search.fit(self.train_X, self.train_y)
			self.model = grid_search.best_estimator_
			self.best_params = grid_search.best_params_
			self.train_score = grid_search.best_score_
		else:
			self.model.fit(self.train_X, self.train_y)
			self.train_score = self.model.score(self.train_X, self.train_y)

	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initilized before predicting.")
		X_test = self.scaler.transform(X)
		return self.model.predict(X_test)

	def print_model_info(self):
		string = ""
		if self.best_params:
			string = f"Model information:\n  Best Parameters: {self.best_params}\n"
		if self.train_score:
			string += f"  Train Score (R^2): {self.train_score:.4f}\n"
		if self.test_score:
			string += f"  Test Score (R^2): {self.test_score:.4f}\n"
		return string
	
	def save(self, file_path):
		if self.model.get_params():
			self.description += f"{self.model.get_params()}"
		with open(file_path, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def load(file_path):
		with open(file_path, "rb") as f:
			return pickle.load(f)
