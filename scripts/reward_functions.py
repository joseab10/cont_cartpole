import numpy as np
from continuous_cartpole import angle_normalize


def rf_inf_generator(time_steps, cos_pow=3, max_pts=100):

	def rf_inf(cart_pole):
		""" Informative Reward Function:
		This Reward Function returns a float reward according to:

				⎧ -<max_pts>                            if x ≤ -1 or 1 ≤ x
			r = ⎨
				⎪                       <cos_pow>
				⎪ __<max_pts>__ * cos(θ)^              else
				⎩  <time_steps>

		where:
			· x is the horizontal position of the car in [-1, 1]
			· θ is the angular position of the pole (θ=0 upwards)

		:param cart_pole: CartPole Environment from OpenAI Gym
		:return: (float) reward in interval [-<max_pts>, <max_pts>/<time_steps>]
		"""

		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -max_pts
		else:
			return (np.cos(cart_pole.state[2])**cos_pow) * (max_pts / time_steps)

	return rf_inf


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
	return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 0


def rf_info_pos(cart_pole):
	""" Sparse positive Reward Function:
	This Reward Function returns a positive reward in the interval [0, 1] given by:

		r = (cos(θ) + 1) / 2

	where:
		· θ is the angle of the pole (θ=0 upwards)

	:param cart_pole: CartPole Environment from OpenAI Gym
	:return: (float) reward between [0, 1]
	"""
	return (np.cos(cart_pole.state[2]) + 1) / 2
