import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.scale as mscale

class ReciprocalFormatter(ticker.FuncFormatter):
	def __init__(self, offset=None):
		self.offset = offset
		self.offset_string = ''

	def __call__(self, x, _):
		if x == 0:
			return "∞"
		val = 1.0 / x
		if self.offset:
			val /= self.offset
		return f"{val:.4g}"

class ReciprocalLocator(ticker.Locator):
	def __init__(self, *, numticks=None):
		self.numticks = numticks

	def set_params(self, *, numticks=None):
		if numticks is not None:
			self.numticks = numticks
	
	def __call__(self):
		vmin, vmax = self.axis.get_view_interval()
		return self.tick_values(vmin, vmax)

	def tick_values(self, vmin, vmax):
#		if vmin < 1E-30 or vmax < 1E-30:
#			raise ValueError(f"Reciprocal scaling requires positive limits, vmin={vmin} and vmax={vmax}")
		recip_vmin, recip_vmax = 1.0 / vmax, 1.0 / vmin
		if self.numticks is None and self.axis is not None:
			self.numticks = self.axis.get_tick_space()
		locator = ticker.MaxNLocator(self.numticks)	
		reciprocal_ticks = locator.tick_values(recip_vmin, recip_vmax)
		return 1.0 / reciprocal_ticks

class ReciprocalScale(mscale.ScaleBase):
	name = 'reciprocal'

	def __init__(self, axis, **kwargs):
		data_min, data_max = axis.get_data_interval()
		if data_min <= 0:
			data_min = 1e-30
		if data_max <= 0:
			raise ValueError("Reciprocal scaling is only provided for positive data")
		reciprocal_range = 1.0 / np.array([data_max, data_min])
		max_reciprocal = np.max([np.abs(reciprocal_range)])
		self.offset = 10 ** np.floor(np.log10(max_reciprocal))
		print(f"offset {self.offset}")

	def get_transform(self):
		return self.ReciprocalTransform()

	def set_default_locators_and_formatters(self, axis):
		axis.set_major_locator(ReciprocalLocator())
		axis.set_major_formatter(ReciprocalFormatter(self.offset))
#		axis.set_minor_locator(ReciprocalLocator())
		self.adjust_offset(axis)

	class ReciprocalTransform(mscale.Transform):
		input_dims = output_dims = 1

		def transform_non_affine(self, a):
			return -1.0 / np.array(a)

		def inverted(self):
			return ReciprocalScale.InvertedReciprocalTransform()

	class InvertedReciprocalTransform(mscale.Transform):
		input_dims = output_dims = 1

		def transform_non_affine(self, a):
			return -1.0 / np.array(a)

		def inverted(self):
			return ReciprocalScale.ReciprocalTransform()

	def adjust_offset(self, axis):
		if self.offset:
			exponent = int(np.log10(self.offset))
			axis.get_offset_text().set_visible(False)
			label = axis.get_label().get_text()
			axis.set_label_text(f"10^{-exponent} / {label}")
