from continuous_cartpole import ContinuousCartPoleEnv, angle_normalize

from continuous_reinforce import REINFORCE, GaussPolicy, BetaPolicy

from utils import plot_episode_stats

import numpy as np



if __name__ == "__main__":

	# episodes = 3000
	episodes = 10000
	time_steps = 500


	def informative_reward(cart_pole):

		cos_pow = 3
		max_pts = 100

		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -max_pts
		else:
			return (np.cos(cart_pole.state[2])**cos_pow)*(max_pts/(2 * time_steps))




	scenarios = [
		#{'desc': 'REINFORCE without Baseline', 'bl': False, 'suffix': 'no_bl'},
		{'desc': 'REINFORCE with Baseline'   , 'bl': False , 'suffix': 'w_bl', 'rewardF': informative_reward},
		#{'desc': 'REINFORCE with Baseline'   , 'bl': True , 'suffix': 'w_bl', 'rewardF': None}
	]



	for scenario in scenarios:

		#num_runs = 10

		#runs_stats = []
		#for _ in range(num_runs):

		print()
		print(scenario['desc'])

		env = ContinuousCartPoleEnv(reward_function=scenario['rewardF'])
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]

		reinforce = REINFORCE(env, state_dim, action_dim, gamma=0.99)

		stats = reinforce.train(episodes, time_steps, baseline=scenario['bl'])

		plot_episode_stats(stats, scenario['suffix'])

		for _ in range(5):
			s = env.reset()
			for _ in range(500):
				env.render()
				a = reinforce.get_action(s)
				s, _, d, _ = env.step(a)
				if d:
					break

		env.viewer.close()