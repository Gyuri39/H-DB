from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from .ml_model import MachineLearningModel

class GaussianProcessRegressionModel(MachineLearningModel):
	def __init__(self):
		super().__init__(name="GPR", description = "Gaussian process regression")
		self._yes_dev = True
		self.kernel = RBF()
		self.multiple_char_lengths = False
		self.alpha = 1e-10
		self.n_restarts = 0
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts, random_state=0, normalize_y=True)
		

	def set_RBFkernel(self, multipleRBF, white_kernel):
		if self.train_X is None:
			raise NotImplementedError("A training set must be initialized before scaling.")
		if multipleRBF == True:
			lengths = [1.0] * self.train_X.shape[0]
			if white_kernel == True:
				self.kernel = RBF(length_scale=lengths) + WhiteKernel()
			elif white_kernel == False:
				self.kernel = RBF(length_scale=lengths)
			else:
				raise KeyError("Invalid white kernel designation")
		elif multipleRBF == False:
			if white_kernel == True:
				self.kernel = RBF(length_scale=1.0) + WhiteKernel()
			elif white_kernel == False:
				self.kernel = RBF(length_scale=1.0)
			else:
				raise KeyError("Invalid white kernel designation")
		else:
			raise KeyError("Invalid multiple characteristic length designation")
		self.model = GaussianProcessRegressor(kernel = self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts, random_state=0, normalize_y=True)

	def train(self):
		super().train()


	def predict(self, X):
		if self.model is None:
			raise NotImplementedError("A model must be initialized before predicting.")
		X_test = self.scaler.transform(X)
		return self.model.predict(X_test, return_std=True)

