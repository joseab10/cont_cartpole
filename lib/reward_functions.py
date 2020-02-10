import numpy as np
from continuous_cartpole import angle_normalize



def rf_inf(cart_pole, cos_pow=3):
	""" Informative Reward Function:
	This Reward Function returns a float reward according to:

			⎧       <cos_pow>
		r = ⎨ cos(θ)^           if -2.4 ≤ x ≤ 2.4
			⎪
			⎩  -1               else

	where:
		· x is the horizontal position of the car
		· θ is the angular position of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (float) reward in interval [-1, 1]
	"""

	x = cart_pole.state[0]
	theta = angle_normalize(cart_pole.state[2])

	if -cart_pole.x_threshold <= x <= cart_pole.x_threshold:
		return np.cos(theta) ** cos_pow
	else:
		return -1



def rf_spar_pos(cart_pole):
	""" Sparse positive Reward Function:
	This Reward Function returns +1 when the pole is within the desired threshold, else it returns 0

			⎧ 1   if -0.1 ≤ θ ≤ 0.1
		r = ⎨
			⎩ 0   else

	where:
		· θ is the angle of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (int) reward in {0, +1}
	"""

	theta = angle_normalize(cart_pole.state[2])

	return 1 if -0.1 <= theta <= 0.1 else 0


def rf_info2d_pos(cart_pole):
	""" Sparse positive Reward Function:
	This Reward Function returns a positive reward in the interval [0, 1] given by:

			⎧ ¼(cos(θ) + 1)(cos(πx) + 1)   if -2.4 ≤ x ≤ 2.4
		r = ⎨
			⎩ 0   else

	where:
		· x is the horizontal position of the car in [-1, 1]
		· θ is the angle of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (float) reward between [0, 1]
	"""
	x = cart_pole.state[0]
	theta = angle_normalize(cart_pole.state[2])

	if -cart_pole.x_threshold <= x <= cart_pole.x_threshold:
		return (np.cos(theta) + 1) * (np.cos(np.pi * x / cart_pole.x_threshold) + 1) / 4
	else:
		return 0


def rf_info2d_sharp_pos(cart_pole):
	""" Sparse positive Reward Function:
	This Reward Function returns a positive reward in the interval [0, 1] given by:

			⎧ ½(cos(θ/2)^17)(cos(πx) + 1)   if -2.4 ≤ x ≤ 2.4
		r = ⎨
			⎩ 0   else

	where:
		· x is the horizontal position of the car in [-1, 1]
		· θ is the angle of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (float) reward between [0, 1]
	"""

	x = cart_pole.state[0]
	theta = angle_normalize(cart_pole.state[2])

	if -cart_pole.x_threshold <= x <= cart_pole.x_threshold:
		return (np.cos(theta/2)**17) * (np.cos(np.pi * x / cart_pole.x_threshold) + 1) / 2
	else:
		return 0


def rf_info_pos(cart_pole):
	""" Sparse positive Reward Function:
	This Reward Function returns a positive reward in the interval [0, 1] given by:

		r = ½(cos(θ) + 1)

	where:
		· θ is the angle of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (float) reward between [0, 1]
	"""
	theta = angle_normalize(cart_pole.state[2])

	return (np.cos(theta) + 1) / 2


'''
	Class Tests
'''
if __name__ == '__main__':

	from collections import namedtuple
	from matplotlib import pyplot as plt

	env_state_threshold = 2.4

	x0 = -3
	x1 = 3
	x_samples = 100

	th0 = -np.pi
	th1 = np.pi
	th_samples = 100

	x_in = np.linspace(x0, x1, x_samples)
	th_in = np.linspace(th0, th1, th_samples)

	time_steps = 500

	def rf_default(cart_pole):
		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -1
		return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 0


	TestCartPole = namedtuple("Env", ["state", "x_threshold"])



	reward_functions = [
		{'rf': rf_default         , 'label': 'Default Reward Function'},
		{'rf': rf_inf             , 'label': 'Cos3 Informative Reward Function'},
		{'rf': rf_spar_pos        , 'label': 'Sparse Positive Reward Function'},
		{'rf': rf_info_pos        , 'label': 'Informative Positive Reward Function'},
		{'rf': rf_info2d_pos      , 'label': '2D Informative Positive Reward Function'},
		{'rf': rf_info2d_sharp_pos, 'label': '2D Sharp Informative Positive Reward Function'}
	]

	num_cols = np.ceil(np.sqrt(len(reward_functions)))
	num_rows = np.ceil(len(reward_functions) / num_cols)

	for p, rf in enumerate(reward_functions):

		r = np.zeros((x_samples, th_samples))

		for i in range(x_samples):
			for j in range(th_samples):
				state = np.array([x_in[i], 0, th_in[j], 0])
				env = TestCartPole(state=state, x_threshold=env_state_threshold)
				r[j, i] = rf['rf'](env)

		ax = plt.subplot(num_rows, num_cols, p + 1, aspect=(x1-x0)/(th1-th0))
		plt.xlabel('x', fontsize=9)
		plt.ylabel('θ', fontsize=9)
		plt.title(rf['label'], fontsize=9)
		ax.tick_params(labelsize=8)

		levels = 40
		cf = plt.contourf(x_in, th_in, r, levels=levels)
		cl = plt.contour(x_in, th_in, r, levels=levels, colors='k', linewidths=0.25)

		cbar = plt.colorbar(cf, format='%.3f')
		cbar.ax.tick_params(labelsize=6)

	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	plt.show()

