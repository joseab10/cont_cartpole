from continuous_cartpole import ContinuousCartPoleEnv, angle_normalize

from continuous_reinforce import *
from deep_q_learning import *

from policies import *
from action_functions import *

from utils import plot_episode_stats

import pickle

import numpy as np



if __name__ == "__main__":

	runs       =   10       # For the purpose of collecting more data given the stochastic nature of the algorithms
	episodes   = 5000 #5000 # Number of training episodes per run
	time_steps =  500 #500  # Number of maximum time steps per episode

	test_episodes = 5 # Number of testing episodes per run


	state_dim = 4
	cont_act_dim = 1
	disc_act_dim = 2

	gamma = 0.99

	visualize = False

	# Upright starting position
	initial_state = np.zeros(4)

	model_dir = '../save/models'
	plt_dir   = '../save/plots'
	data_dir  = '../save/data'


	# Function Definitions
	def informative_reward(cart_pole):

		cos_pow = 3
		max_pts = 100

		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -max_pts
		else:
			return (np.cos(cart_pole.state[2])**cos_pow)*(max_pts/(2 * time_steps))


	def clip_action(a):
		return np.clip(a, -1, 1)


	# Value Functions
	Q = StateActionValueFunction(state_dim, disc_act_dim)
	Q_Target = StateActionValueFunction(state_dim, disc_act_dim)

	V = StateValueFunction(state_dim)

	# Policies
	beta_policy = BetaPolicy(state_dim, cont_act_dim, act_min=-1, act_max=1)
	epsilon_greedy = EpsilonGreedyPolicy(0.2, Q)

	# Action Preprocessing functions
	# Convert from discrete to continuous actions through a dictionary mapping
	act_disc2cont = ActDisc2Cont({0: -1, 1: 1})
	# Clips actions between [-1, 1]
	act_clipcont  = ActCont2Cont(clip_action, clip_action)


	# Algorithms
	dqn_algorithm = DQN(epsilon_greedy, act_disc2cont, Q, Q_Target, state_dim, disc_act_dim, gamma=gamma, doubleQ=True)
	reinforce     = REINFORCE(beta_policy, act_clipcont, state_dim, cont_act_dim, gamma=gamma)
	reinforce_wbl = REINFORCE(beta_policy, act_clipcont, state_dim, cont_act_dim, gamma=gamma, baseline = True, V=V)



	scenarios = [
		{'exe': False, 'desc': 'REINFORCE without Baseline'     , 'alg': reinforce    , 'suffix': 'no_bl'                  , 'rewardF': None              , 'init': None},
		{'exe': False,  'desc': 'REINFORCE with BL, OF, and Init', 'alg': reinforce_wbl, 'suffix': 'cont_rfc_bl_of_in_beta_', 'rewardF': informative_reward, 'init': initial_state},
		{'exe': True,  'desc': 'DQN'                            , 'alg': dqn_algorithm, 'suffix': 'disc_dqn'               , 'rewardF': None              , 'init': None},
		{'exe': True,  'desc': 'DQN with Info ObjectiveFunc'    , 'alg': dqn_algorithm, 'suffix': 'disc_dqn_of'            , 'rewardF': informative_reward, 'init': None}
	]



	for scenario in scenarios:

		if scenario['exe']:

			print('\n\n\n***' + scenario['desc'])

			run_train_stats = []
			run_test_stats = []


			for run in range(runs):

				print('\n** RUN {}'.format(run + 1))
				print('* Training')

				# Training
				env = ContinuousCartPoleEnv(reward_function=scenario['rewardF'])

				state_dim = env.observation_space.shape[0]
				action_dim = env.action_space.shape[0]

				algorithm = scenario['alg']
				stats = algorithm.train(env, episodes, time_steps, initial_state=scenario['init'])

				run_train_stats.append(stats)

				algorithm.save(model_dir, 'model_{}_run_{}'.format(scenario['suffix'], run))

				print('\n* Testing')
				test_stats = EpisodeStats(episode_lengths=np.zeros(test_episodes), episode_rewards=np.zeros(test_episodes))
				for e in range(test_episodes):
					s = env.reset(initial_state=scenario['init'])

					for t in range(time_steps):

						if visualize:
							env.render()

						a = algorithm.get_action(s, deterministic=True)
						s, r, d, _ = env.step(a)

						test_stats.episode_rewards[e] += r
						test_stats.episode_lengths[e] = t

						if d:
							break

					print("{} Steps in Episode {}/{}. Reward {}".format(t + 1, e + 1, test_episodes, test_stats.episode_rewards[e]))

				run_test_stats.append(test_stats)

				if visualize:
					env.viewer.close()

			# Aggregate stats for all runs
			train_rewards = []
			train_lengths = []
			test_rewards  = []
			test_lengths  = []

			for r in range(runs):
				train_rewards.append(run_train_stats[r].episode_rewards)
				train_lengths.append(run_train_stats[r].episode_lengths)
				test_rewards.append(run_test_stats[r].episode_rewards)
				test_lengths.append(run_test_stats[r].episode_lengths)

			train_rewards = np.array(train_rewards)
			train_lengths = np.array(train_lengths)
			test_rewards  = np.array(test_rewards)
			test_lengths  = np.array(test_lengths)

			train_length_means = np.mean(train_lengths, axis=1)
			train_length_stdev =  np.std(train_lengths, axis=1)
			train_reward_means = np.mean(train_rewards, axis=1)
			train_reward_stdev =  np.std(train_rewards, axis=1)

			test_length_means = np.mean(test_lengths, axis=1)
			test_length_stdev =  np.std(test_lengths, axis=1)
			test_reward_means = np.mean(test_rewards, axis=1)
			test_reward_stdev =  np.std(test_rewards, axis=1)


			plot_stats=[{'run': 'train',
						 'reward_means': train_reward_means, 'reward_stdev': train_reward_stdev, 'rewards': train_rewards,
						 'length_means': train_length_means, 'length_stdev': train_length_stdev, 'lengths': train_lengths},
						{'run': 'test' ,
						 'reward_means': test_reward_means, 'reward_stdev': test_reward_stdev, 'rewards': test_rewards,
						 'length_means': test_length_means, 'length_stdev': test_length_stdev, 'lengths': test_lengths}]

			with open(data_dir + '/data' + scenario['suffix'] + '_' + timestamp() + '.pkl', 'wb') as f:
				pickle.dump(plot_stats, f)

			plot_run_stats(plot_stats, dir=plt_dir, suffix=scenario['suffix'], noshow=not visualize)


