import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from firebase_admin import firestore
import joblib
import base64
from io import BytesIO, StringIO
import importlib

MODEL_MODULES = {
	"LinearRegressionModel": "server.models.linear_regression_model",
	"ElasticNetModel": "server.models.linear_regression_model",
	"GaussianProcessRegressionModel": "server.models.gaussian_process_model"
}

class BaseRegressionModel:
	def __init__(self, name, description=""):
		self.class_name = self.__class__.__name__
		self.name = name
		self.description = description
		self.model = None
		self.train_X = None
		self.train_y = None
		self.train_Xstd = None
		self.train_ystd = None
		self.scaler = None
		self._yes_dev = False
		self.sample_weight = None
		self.training_info = None
		self.description_others = None
		self._yes_gridcv = False
		self.param_grid = None
		self.grid_search = None
		self.best_params = None
		self.train_score = None
		self.test_score = None
		self.feature_transform = None
		self.target_transform = None

	def apply_transformation(self, data, transform_types):
		transformed_data = data.copy()
		if isinstance(transformed_data, pd.Series):
			transformed_data = transformed_data.to_frame()
		for i, transform_type in enumerate(transform_types):
			if transform_type == "linear":
				continue
			elif transform_type == "log":
				transformed_data.iloc[:,i] = np.log(transformed_data.iloc[:,i])
			elif transform_type == "log10":
				transformed_data.iloc[:,i] = np.log10(transformed_data.iloc[:,i])
			elif transform_type == "reciprocal":
				transformed_data.iloc[:,i] = 1.0/transformed_data.iloc[:,i]
			else:
				raise ValueError(f"Unsupported transformation: {transform_type}")
		return transformed_data

	def inverse_transformation(self, data, transform_types):
		transformed_data = data.copy()
		if isinstance(transformed_data, np.ndarray):
			transformed_data = pd.DataFrame(transformed_data)
		if isinstance(transformed_data, pd.Series):
			transformed_data = transformed_data.to_frame()
		for i, transform_type in enumerate(transform_types):
			if transform_type == "linear":
				continue
			elif transform_type == "log":
				transformed_data.iloc[:,i] = np.exp(transformed_data.iloc[:,i])
			elif transform_type == "log10":
				transformed_data.iloc[:,i] = np.power(10, transformed_data.iloc[:,i])
			elif transform_type == "reciprocal":
				transformed_data.iloc[:,i] = 1.0/transformed_data.iloc[:,i]
			else:
				raise ValueError(f"Unsupported transformation: {transform_type}")
		return transformed_data

	def apply_error_transformation(self, data, errdata, transform_types):
		transformed_errdata = errdata.copy()
		if isinstance(transformed_errdata, pd.Series):
			transformed_errdata = transformed_errdata.to_frame()
		if isinstance(data, pd.Series):
			data = data.to_frame()
		for i, transform_type in enumerate(transform_types):
			if transform_type == "linear":
				continue
			elif transform_type == "log":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] / abs(data.iloc[:,i])
			elif transform_type == "log10":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] / (abs(data.iloc[:,i]) * np.log(10))
			elif transform_type == "reciprocal":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] / (data.iloc[:,i]**2)
			else:
				raise ValueError(f"Unsupported transformation: {transform_type}")
		return transformed_errdata
	
	def inverse_error_transformation(self, data, errdata, transform_types):
		transformed_errdata = errdata.copy()
		if isinstance(transformed_errdata, np.ndarray):
			transformed_errdata = pd.DataFrame(transformed_errdata)
		if isinstance(transformed_errdata, pd.Series):
			transformed_errdata = transformed_errdata.to_frame()
		if isinstance(data, np.ndarray):
			data = pd.DataFrame(data)
		if isinstance(data, pd.Series):
			data = data.to_frame()
		for i, transform_type in enumerate(transform_types):
			if transform_type == "linear":
				continue
			elif transform_type == "log":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] * abs(data.iloc[:,i])
			elif transform_type == "log10":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] * abs(data.iloc[:,i]) * np.log(10)
			elif transform_type == "reciprocal":
				transformed_errdata.iloc[:,i] = transformed_errdata.iloc[:,i] * (data.iloc[:,i] ** 2)
			else:
				raise ValueError(f"Unsupported transformation: {transform_type}")
		return transformed_errdata

	def set_training_info(self, description:str):
		self.training_info = description

	def set_description_others(self, description:str):
		self.description_others = description

	def set_data(self, train_X: pd.DataFrame, train_y: pd.DataFrame):
		self.train_X = train_X
		self.train_y = train_y
	
	def set_transform(self, feature_transform=None, target_transform="linear"):
		if feature_transform is None:
			feature_transform = ["linear"] * self.train_X.shape[1]
		if len(feature_transform) != self.train_X.shape[1]:
			raise ValueError("len(feature_transform) is not equal to the number of features.")
		self.feature_transform = feature_transform
		self.target_transform = [target_transform]
		
	def set_scaler(self, scaler_name):
		if self.train_X is None:
			raise NotImplementedError("A training set must be initialized before scaling.")
		elif self.feature_transform is None:
			raise NotImplementedError("Feature transform must be initialized before scaling.")
		else:
			X_transformed = self.apply_transformation(self.train_X, self.feature_transform)

		if scaler_name.startswith("Standard"):	#Standardize
			self.scaler = StandardScaler().fit(X_transformed)
		elif scaler_name.startswith("Normal"):	#Min-Max Scale or Normalize
			self.scaler = MinMaxScaler().fit(X_transformed)
		elif scaler_name == "None": #No Scaling
			scaler = StandardScaler().fit(X_transformed)
			scaler.mean_ = np.zeros(X_transformed.shape[1])
			scaler.scale_ = np.ones(X_transformed.shape[1])
			self.scaler = scaler
		else:
			raise KeyError("Invalid scaler import trial")

	def set_sample_weight(self, sample_weight):
		self.sample_weight = sample_weight

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
		X_train = self.scaler.transform(self.apply_transformation(self.train_X, self.feature_transform))
		y_train = self.apply_transformation(self.train_y, self.target_transform)
		if self._yes_gridcv:
			grid_search.fit(self.train_X, self.train_y)
			self.model = grid_search.best_estimator_
			self.best_params = grid_search.best_params_
			self.train_score = grid_search.best_score_
		else:
			self.model.fit(X_train, y_train)
			self.train_score = self.model.score(X_train, y_train)

	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initilized before predicting.")
		X_test = self.scaler.transform(self.apply_transformation(X, self.feature_transform))
		y_pred = self.model.predict(X_test)
		return self.inverse_transformation(y_pred, self.target_transform)

	def print_model_info(self):
		string = ""
		if self.best_params:
			string = f"Model information:\n  Best Parameters: {self.best_params}\n"
		if self.train_score:
			string += f"  Train Score (R^2): {self.train_score:.4f}\n"
		if self.test_score:
			string += f"  Test Score (R^2): {self.test_score:.4f}\n"
		return string
	
	def serialize_model(self):
		model_buffer = BytesIO()
		joblib.dump(self.model, model_buffer)
		model_bytes = model_buffer.getvalue()
		return base64.b64encode(model_bytes).decode('utf-8')
	
	def serialize_scaler(self):
		scaler_buffer = BytesIO()
		joblib.dump(self.scaler, scaler_buffer)
		scaler_bytes = scaler_buffer.getvalue()
		return base64.b64encode(scaler_bytes).decode('utf-8')

	def to_dict(self):
		def df2csv(df):
			if df is not None:
				csv_buffer = StringIO()
				df.to_csv(csv_buffer, index=False)
				return csv_buffer.getvalue()
			return None
		return {
			'class': self.class_name,
			'name': self.name,
			'description': self.description,
			'feature_transform': self.feature_transform,
			'target_transform': self.target_transform,
			'model': self.serialize_model(),
			'train_X': df2csv(self.train_X),
			'train_y': df2csv(self.train_y),
			'train_Xstd': df2csv(self.train_Xstd),
			'train_ystd': df2csv(self.train_ystd),
			'scaler': self.serialize_scaler(),
			'_yes_dev': self._yes_dev,
			'training_info': self.training_info,
			'description_others': self.description_others,
			'_yes_gridcv': self._yes_gridcv,
			'param_grid': self.param_grid,
			'grid_search': self.grid_search,
			'best_params': self.best_params,
			'train_score': self.train_score,
			'test_score': self.test_score
		}

	def save(self, document_id, collection_name="models"):
		if self.model.get_params():
			self.description += f"{self.model.get_params()}"

		data = self.to_dict()
		db_client = firestore.client()
		doc_ref = db_client.collection(collection_name).document(document_id)
		doc_ref.set(data)

	@classmethod
	def deserialize_model(cls, model_base64):
		if model_base64:
			model_bytes = base64.b64decode(model_base64.encode('utf-8'))
			model_buffer = BytesIO(model_bytes)
			return joblib.load(model_buffer)
		return None

	@classmethod
	def deserialize_scaler(cls, scaler_base64):
		if scaler_base64:
			scaler_bytes = base64.b64decode(scaler_base64.encode('utf-8'))
			scaler_buffer = BytesIO(scaler_bytes)
			return joblib.load(scaler_buffer)
		return None

	@classmethod
	def dictload(cls, document_id, collection_name="models"):
		db_client = firestore.client()
		doc_ref = db_client.collection(collection_name).document(document_id)
		doc = doc_ref.get()

		def csv2df(csv_name):
			try:
				df = pd.read_csv(StringIO(data.get(csv_name)))
				return df
			except:
				return None

		if doc.exists:
			data = doc.to_dict()
			model_class_name = data.get('class', None)
			model_class = cls
			if model_class_name:
				try:
					module_name = MODEL_MODULES[model_class_name]
					module = importlib.import_module(module_name)
					model_class = getattr(module, model_class_name, cls)
				except Exception as e:
					print(f"Error loading class {model_class_name}: {e}")

			data['model'] = cls.deserialize_model(data.get('model'))
			data['scaler'] = cls.deserialize_scaler(data.get('scaler'))
			for df_name in ['train_X', 'train_y', 'train_Xstd', 'train_ystd']:
				data[df_name] = csv2df(df_name)
			data['_model_class'] = model_class if model_class and issubclass(model_class, BaseRegressionModel) else cls
			return data
		else:
			return None
		
	@classmethod
	def load(cls, document_id, collection_name="models"):
		dict_data = cls.dictload(document_id, collection_name) if cls != BaseRegressionModel else BaseRegressionModel.dictload(document_id)
		if dict_data is None:
			print("No data returned from dictload")
			return None

		model_class = dict_data.pop('_model_class', cls)

		if model_class != cls:
			dict_data = model_class.dictload(document_id, collection_name)
			if dict_data is None:
				print("No data returned from subclass dictload")
				return None
		instance = model_class(name=dict_data['name'])

		for key, value in dict_data.items():
			if hasattr(instance, key):
				setattr(instance, key, value)
		return instance
