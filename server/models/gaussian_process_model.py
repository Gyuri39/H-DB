from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from .base_model import BaseRegressionModel
import streamlit as st
from io import BytesIO
import joblib
import base64

class GaussianProcessRegressionModel(BaseRegressionModel):
	def __init__(self):
		super().__init__(name="GPR", description = "Gaussian process regression")
		self._yes_dev = True
		self.kernel = RBF()
		self.multiple_char_lengths = False
		self.alpha = 1e-10
		self.n_restarts = 0
		self.noise_level_bounds = (1e-5, 1e+5)
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts, random_state=None, normalize_y=True)

	def set_RBFkernel(self, multipleRBF, white_kernel, length_scale = 1.0e-10, length_scale_bounds=(1e-20, 1e+10)):
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
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha = self.alpha, n_restarts_optimizer = self.n_restarts, random_state = None, normalize_y = False)

	def set_alpha(self, noise_level):
		self.alpha = noise_level ** 2
		self._update_model()
	
	def set_noise_bounds(self, noise_min, noise_max):
		self.noise_level_bounds = (noise_min, noise_max)
		self._update_model()

	def train(self):
		super().train()
		self.best_params = f"GPR with kernel {self.model.kernel_}"
		st.warning(self.best_params)

	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before predicting.")
		X_test = self.scaler.transform(X)
		pred_result = self.model.predict(X_test, return_std=True)
		if isinstance(pred_result, tuple) and len(pred_result) == 2:
			return pred_result
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
		ml_dict = super().dictload(document_id)
		if ml_dict:
			data = ml_dict.__dict__.copy()
			data['kernel'] = cls.deserialize_kernel(data.get('kernel'))
			return data
		return None
