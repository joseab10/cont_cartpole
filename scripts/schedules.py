import numpy as np


def sch_no_decay(x, x0, x1, y0, y1):
	""" Constant (no decaying) schedule function

	It returns:

		y = f(x) = y₁


	:param x:  (float, ndarray) Input of the function
	:param x0:   *Unused
	:param x1:   *Unused
	:param y0:   *Unused
	:param y1: (float) Constant output of the function
	:return:   (float, ndarray) y = f(x)
	"""

	del x0, x1, y0  # Unused parameters

	if isinstance(x, np.ndarray):
		return y1 * np.ones_like(x)
	else:
		return y1


def sch_linear_decay(x, x0, x1, y0, y1):
	""" Linear Decay schedule function

	It returns:

		y = f(x) = m·x + b

	where :
			   y₁ - y₀
		· m = ---------
			   x₁ - x₀

		· b = y₀ - m·x₀

	:param x:  (float, ndarray) Input of the function
	:param x0: (float) Starting point of the linear decay along x
	:param x1: (float) Stopping point of the linear decay along x
	:param y0: (float) Starting point of the linear decay along y
	:param y1: (float) Stopping point of the linear decay along y
	:return:   (float, ndarray) y = f(x)
	"""
	m = (y1 - y0) / (x1 - x0)
	b = y0 - m * x0

	return m * x + b


def sch_exp_decay(x, x0, x1, y0, y1):
	""" Exponential Decay schedule function

	It returns:

								 β(x - x₀)
		y = f(x) = (y₀ - α·y₁) e^          + α·y₁

	where:
			  ⎧ 1.02  if y₁ > y₀
		· α = ⎨
			  ⎩ 0.98  else

				  1         ⎛                1 - α    ⎞
		· β = -------- * log⎜ sgn(y₀ - y₁)----------- ⎟
		      x₁ - x₀       ⎝              α·y₁ + y₀  ⎠


	:param x:  (float, ndarray) Input of the function
	:param x0: (float) Starting point of the linear decay along x
	:param x1: (float) Stopping point of the linear decay along x
	:param y0: (float) Starting point of the linear decay along y
	:param y1: (float) Stopping point of the linear decay along y
	:return:   (float, ndarray) y = f(x)
	"""

	# Given that the exponential aproaches asymptotically the final value,
	# this factor is to lower the asymptote a little bit to allow for an exact minimum value of y after t epochs,
	# otherwise, it would just aproximate e_min, but never actually reach it
	alpha = 0.98
	sign = 1
	if y1 > y0:
		alpha = 2 - alpha
		sign = -1

	# Exponential multiplicative factor (e ^ (alpha * x)) to have a nice continuous function (almost tangent to e_min)
	# within the whole training interval while also reaching e_min at the end.
	beta = np.log((sign * (1 - alpha) * y1) / (alpha * y1 + y0)) / (x1 - x0)
	power = (x - x0) * beta

	out = (y0 - (alpha * y1)) * np.exp(power) + (alpha * y1)

	if isinstance(out, np.ndarray):
		if y0 > y1:
			out[out < y1] = y1
		else:
			out[out > y1] = y1

	else:
		if y0 > y1 and out < y1:
			out = y1

	return out


class Schedule:

	def __init__(self, x0, x1, y0, y1, schedule_function,
					# Cosine Annealing Parameters
					cosine_annealing=True, annealing_cycles=3):

		self._x0 = x0
		self._x1 = x1
		self._y0 = y0
		self._y1 = y1

		self._decay_function = schedule_function

		self._cosine_annealing = cosine_annealing
		# Number of full Peaks the cosine will make within the training interval
		self._annealing_cycles = annealing_cycles

		self._step = 0

	def step(self):
		self._step += 1

	def value(self, x):

		y = self._decay_function(x, self._x0, self._x1, self._y0, self._y1)

		# Cosine Annealing
		if self._cosine_annealing:
			# Cosine Amplitude (Height)
			a = (y - self._y1) / 2
			# Cosine Function Centerline Offset (around which the cosine oscillates)
			b = a + self._y1  # = (y + self._y1) / 2

			y = (a * np.cos((2 * self._annealing_cycles + 1) * (np.pi / (self._x1 - self._x0)) * (x - self._x0))) + b

		# X Before and after the decay period
		if isinstance(x, np.ndarray):
			y[np.where(x < self._x0)] = self._y0
			y[np.where(x > self._x1)] = self._y1
		else:
			if x < self._x0:
				y = self._y0
			elif x > self._x1:
				y = self._y1
		return y

	def __call__(self):
		return self.value(self._step)

	def reset_parameters(self):
		self._step = 0


'''
	Class Tests
'''
if __name__ == '__main__':

	from matplotlib import pyplot as plt

	t0 = 50
	t1 = 130

	e0 = 0.90
	e1 = 0.05

	annealing_cycles = 5

	x_in = np.arange(0, 200, 1)

	decay_functions = [
		{'decay': sch_no_decay         , 'annealing': False, 'label': 'Constant'},
		{'decay': sch_linear_decay     , 'annealing': False, 'label': 'Linear'},
		{'decay': sch_exp_decay        , 'annealing': False, 'label': 'Exponential'},

		# Annealed
		{'decay': sch_no_decay         , 'annealing':  True, 'label': 'Constant'},
		{'decay': sch_linear_decay     , 'annealing':  True, 'label': 'Linear'},
		{'decay': sch_exp_decay        , 'annealing':  True, 'label': 'Exponential'},
	]

	for function in decay_functions:
		epsilon = Schedule(t0, t1, e0, e1, schedule_function=function['decay'],
							cosine_annealing=function['annealing'], annealing_cycles=annealing_cycles)

		y = epsilon.value(x_in)

		plt.plot(x_in, y, label=function['label'])

	plt.legend()
	plt.show()

