from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from .base_model import BaseRegressionModel
import streamlit as st
from io import BytesIO
import joblib
import base64

class GaussianProcessRegressionModel(BaseRegressionModel):
	def __init__(self, name=None):
		super().__init__(name="GPR", description = "Gaussian process regression")
		self.class_name = self.__class__.__name__
		self._yes_dev = True
		self.kernel = RBF()
		self.multiple_char_lengths = False
		self.alpha = 1.0
		self.n_restarts = 0
		self.length_scale_bounds = (1e-10, 1e+10)
		self.noise_level_bounds = (1e-10, 1e+10)
		self.normalize_y = True
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts, random_state=None, normalize_y=self.normalize_y)

	def set_RBFkernel(self, multipleRBF, white_kernel, length_scale = 1.0, length_scale_bounds=None):
		if length_scale_bounds == None:
			length_scale_bounds = self.length_scale_bounds
		if self.train_X is None:
			raise NotImplementedError("A training set must be initialized before scaling.")
		if multipleRBF == True:
			lengths = [length_scale] * self.train_X.shape[1]
			if white_kernel == True:
				self.kernel = RBF(length_scale=lengths, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=(self.noise_level_bounds[0]*self.noise_level_bounds[1])**0.5, noise_level_bounds = self.noise_level_bounds)
			elif white_kernel == False:
				self.kernel = RBF(length_scale=lengths, length_scale_bounds=length_scale_bounds)
			else:
				raise KeyError("Invalid white kernel designation")
		elif multipleRBF == False:
			if white_kernel == True:
				self.kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=(self.noise_level_bounds[0]*self.noise_level_bounds[1])**0.5, noise_level_bounds = self.noise_level_bounds)
			elif white_kernel == False:
				self.kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
			else:
				raise KeyError("Invalid white kernel designation")
		else:
			raise KeyError("Invalid multiple characteristic length designation")
		self._update_model()

	def _update_model(self):
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha = self.alpha, n_restarts_optimizer = self.n_restarts, random_state = None, normalize_y = self.normalize_y)

	def set_alpha(self, noise_level):
		self.alpha = noise_level ** 2
		self._update_model()

	def set_length_bounds(self, length_min, length_max):
		self.length_scale_bounds = (length_min, length_max)
#		self._update_model()
	
	def set_noise_bounds(self, noise_min, noise_max):
		self.noise_level_bounds = (noise_min, noise_max)
		self._update_model()
	
	def set_normalize_y(self, normalize_y: bool):
		self.normalize_y = self.normalize_y
		self._update_model()

	def set_n_restarts(self, n_restarts):
		self.n_restarts = n_restarts
		self._update_model()
	
	def train(self):
		super().train()
		self.best_params = f"GPR with kernel {self.model.kernel_}"
		st.warning(self.best_params)
		st.warning(self.model.alpha)
		st.warning(self.train_score)

	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before predicting.")
		X_test = self.scaler.transform(self.apply_transformation(X, self.feature_transform))
		pred_result = self.model.predict(X_test, return_std=True)
		if isinstance(pred_result, tuple) and len(pred_result) == 2:
			returned_prediction = self.inverse_transformation(pred_result[0], self.target_transform)
			#returned_prederr = self.inverse_error_transformation(returned_prediction, pred_result[1], self.target_transform)
			returned_prederr_lower = returned_prediction - self.inverse_transformation(pred_result[0] - pred_result[1], self.target_transform)
			returned_prederr_upper = self.inverse_transformation(pred_result[0] + pred_result[1], self.target_transform) - returned_prediction
			returned_prederr = (returned_prederr_lower, returned_prederr_upper)
			return returned_prediction, returned_prederr
		else:
			raise ValueError(f"Unexpected output from predict(): {pred_result}")

	def serialize_kernel(self):
		kernel_buffer = BytesIO()
		joblib.dump(self.kernel, kernel_buffer)
		kernel_bytes = kernel_buffer.getvalue()
		return base64.b64encode(kernel_bytes).decode('utf-8')

	def to_dict(self):
		base_dict = super().to_dict()
		return {
			**base_dict,
			'alpha': self.alpha,
			'n_restarts': self.n_restarts,
			'kernel': self.serialize_kernel(),
			'multiple_char_lengths': self.multiple_char_lengths
		}
	
	@classmethod
	def deserialize_kernel(cls, kernel_base64):
		if kernel_base64:
			kernel_bytes = base64.b64decode(kernel_base64.encode('utf-8'))
			kernel_buffer = BytesIO(kernel_bytes)
			return joblib.load(kernel_buffer)
		return None
	
	@classmethod
	def dictload(cls, document_id, collection_name="models"):
		base_dict = BaseRegressionModel.dictload(document_id, collection_name)
		if base_dict:
			data = base_dict.copy()
			data['kernel'] = cls.deserialize_kernel(data.get('kernel'))
			return data
		else:
			print("Base_dict not detected")
		return None
