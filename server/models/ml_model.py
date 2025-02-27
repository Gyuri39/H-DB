import pandas as pd
from .base_model import BaseRegressionModel

class MachineLearningModel(BaseRegressionModel):
	def __init__(self, name, description=""):
		super().__init__(name, description)
		self.alpha = 1e-10
		self.n_restarts = 0
		self.model = None

	def set_alpha(self, noise_level):
		self.alpha = noise_level ** 2
	
	def set_restart(self, num_restarts):
		self.num_restarts = num_restarts

