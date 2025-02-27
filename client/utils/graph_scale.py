import numpy as np
from matplotlib.ticker import AutoLocator, FuncFormatter
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase
from matplotlib.ticker import Locator



class ReciprocalScale(ScaleBase):
	name = 'reciprocal'
	
	def __init__(self, axis, **kwargs):
		super().__init__(axis, **kwargs)
		self.offset = None

	def get_transform(self):
		return self.ReciprocalTransform()

	def set_default_locators_and_formatters(self, axis):
		class ReciprocalLocator(Locator):
			def __init__(self, numticks = 5):
				self.numticks = numticks
			def __call__(self):
				vmin, vmax = self.axis.get_view_interval()
				auto_locator = AutoLocator()
				default_ticks = auto_locator.tick_values(vmin, vmax)
				num_ticks = len(default_ticks)
				ticklocs = np.reciprocal(np.linspace(1/vmax, 1/vmin, num_ticks))
				print(f"ticklocs {ticklocs} inverse {np.reciprocal(ticklocs)}")
				return self.raise_if_exceeds(ticklocs)
		axis.set_major_locator(ReciprocalLocator(numticks=12))
	
	class ReciprocalTransform(Transform):
		input_dims = 1
		output_dims = 1
		is_separable = True

		def transform_non_affine(self, a):
			return np.array(a)**(-1)
		def inverted(self):
			return ReciprocalScale.InvertedReciprocalTransform()
	
	class InvertedReciprocalTransform(Transform):
		input_dims = 1
		output_dims = 1
		is_separable = True

		def transform(self, a):
			return np.array(a)**(-1)
		def inverted(self):
			return ReciprocalScale.ReciprocalTransform()
	
	def get_transform(self):
		return self.ReciprocalTransform()

#	def set_default_locators_and_formatters(self, axis):
#		self.adjust_offset(axis)
#		axis.set_major_locator(ReciprocalMajorLocator())
#		axis.set_major_formatter(FuncFormatter(self._format_ticks_with_offset))

	def _format_ticks_with_offset(self,x,pos):
		if self.offset is None:
			return f"{1 / x:.4f}" if x!=0 else np.inf
		else:
			scaled_value = (1 / x) / self.offset if x != 0 else 0
			return f"{scaled_value:.2f}"

	def adjust_offset(self, axis):
		data_min, data_max = axis.get_data_interval()
		if data_min == 0:
			data_min = 1e-10
		reciprocal_range = 1 / np.array([data_max, data_min])
		max_reciprocal = np.max(np.abs(reciprocal_range))
		self.offset = 10 ** np.floor(np.log10(max_reciprocal))
		label = axis.get_label().get_text()
		axis.set_label_text(f"{label} x10^{int(np.log10(self.offset))}")

#	class ReciprocalTransform(Transform):
#		input_dims = 1
#		output_dims = 1
#		is_separable = True
	
#		def transform(self, values):
#			values = np.array(values, dtype=np.float64)
#			with np.errstate(divide='ignore', invalid='ignore'):
#				result = np.where(values !=0, 1 / values, np.inf)
#			return result

#		def inverted(self):
#			return ReciprocalScale.ReciprocalInverseTransform()

#	class ReciprocalInverseTransform(Transform):
#		input_dims = 1
#		output_dims = 1
#		is_separable = True
#
#		def transform(self, values):
#			values = np.array(values, dtype=np.float64)
#			with np.errstate(divide='ignore', invalid='ignore'):
#				result = np.where(values !=0, 1 / values, np.inf)
#			return result

#		def inverted(self):
#			return ReciprocalScale.ReciprocalTransform()
#
#class ReciprocalMajorLocator(Locator):
#	def __init__(self, num_ticks=5):
#		self.num_ticks = num_ticks
#	def __call__(self):
#		vmin, vmax = self.axis.get_view_interval()
#		if vmin == 0:
#			vmin = 1e-16
#		if vmax == 0:
#			vmax = 1e-16
#		reciprocal_min = 1 / vmax
#		reciprocal_max = 1 / vmin
#		auto_locator = AutoLocator()
#		num_ticks = len(auto_locator)
#		reciprocal_ticks = np.linspace(reciprocal_min, reciprocal_max, num_ticks)
#		return 1/ reciprocal_ticks
