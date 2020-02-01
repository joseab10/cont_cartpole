from continuous_cartpole import ContinuousCartPoleEnv, angle_normalize

from continuous_reinforce import *
from deep_q_learning import *

from policies import *
from action_functions import *

from utils import *

import pickle

import numpy as np

from parameters import load_parameters

import argparse



def test_agent(env, agent, episodes=5, time_steps=500, initial_state=None, initial_noise=None, render=True):

	stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes), episode_loss=np.zeros(episodes))

	print_header(3, 'Testing')

	for e in range(episodes):

		s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)

		for t in range(time_steps):

			if render:
				env.render()

			a = agent.get_action(s, deterministic=True)
			s, r, d, _ = env.step(a)

			stats.episode_rewards[e] += r
			stats.episode_lengths[e] = t

			if d:
				break

		pr_stats = {'steps': int(stats.episode_lengths[e] + 1), 'episode': e + 1, 'episodes': episodes,
					'reward': stats.episode_rewards[e]}
		print_stats(pr_stats)

	if render:
		env.viewer.close()

	return stats





if __name__ == "__main__":


	#runs       =   3 # 10 For the purpose of collecting more data given the stochastic nature of the algorithms
	#episodes   = 10 # 5000 Number of training episodes per run
	#time_steps =  200 # 500 Number of maximum time steps per episode

	#test_episodes = 5 # Number of testing episodes per run


	#state_dim = 4
	#cont_act_dim = 1
	#disc2_act_dim = 2
	#disc4_act_dim = 2

	#gamma = 0.99

	#visualize = True

	# Upright starting position
	#initial_state = np.zeros(4)
	# Downwards starting position
	#initial_state = np.array([0., 0., np.pi, 0.])

	# Little starting noise
	#initial_noise = 0.5
	# 360º starting noise for angular position
	#initial_noise = np.array([0.75, 0.5, np.pi, 0.5])




	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', action='store', default='experiments_example.json', help='Path to the experiments file.', type=str)

	args = parser.parse_args()

	# Value Functions
	#Q = StateActionValueFunction(state_dim, disc2_act_dim)
	#Q_Target = StateActionValueFunction(state_dim, disc2_act_dim)

	#Q4 = StateActionValueFunction(state_dim, disc4_act_dim)
	#Q4_Target = StateActionValueFunction(state_dim, disc4_act_dim)

	#V = StateValueFunction(state_dim)

	# Policies
	#beta_policy = BetaPolicy(state_dim, cont_act_dim, act_min=-1, act_max=1)
	#epsilon_greedy = EpsilonGreedyPolicy(0.2, Q)
	#epsilon_greedy4 = EpsilonGreedyPolicy(0.2, Q4)

	# Action Preprocessing functions



	# Algorithms
	#dqn_algorithm = DQN(epsilon_greedy, act_disc2cont, Q, Q_Target, state_dim, disc2_act_dim, gamma=gamma, doubleQ=True)
	#dqn4_algorithm = DQN(epsilon_greedy4, act_disc4cont, Q4, Q4_Target, state_dim, disc4_act_dim, gamma=gamma, doubleQ=True)
	#reinforce     = REINFORCE(beta_policy, act_clipcont, state_dim, cont_act_dim, gamma=gamma)
	#reinforce_wbl = REINFORCE(beta_policy, act_clipcont, state_dim, cont_act_dim, gamma=gamma, baseline = True, V=V)



	#scenarios = [
	#	{'exe': False, 'desc': 'REINFORCE without Baseline'     , 'alg': reinforce    , 'suffix': 'no_bl'                  , 'rewardF': None              , 'init': None, 'noise': None},
	#	{'exe': False,  'desc': 'REINFORCE with BL, OF, and Init', 'alg': reinforce_wbl, 'suffix': 'cont_rfc_bl_of_in_beta_', 'rewardF': informative_reward, 'init': initial_state, 'noise': None},
	#	{'exe': False,  'desc': 'DQN'                            , 'alg': dqn_algorithm, 'suffix': 'disc_dqn'               , 'rewardF': None              , 'init': None, 'noise': None},
	#	{'exe': False, 'desc': 'DQN', 'alg': dqn_algorithm, 'suffix': 'disc2_dqn_nis', 'rewardF': None, 'init': None, 'noise': initial_noise}, # Best so far
	#	{'exe': False, 'desc': 'DQN4', 'alg': dqn4_algorithm, 'suffix': 'disc4_dqn_nis', 'rewardF': None, 'init': None,
	#	 'noise': initial_noise},
	#	{'exe': True,  'desc': 'DQN with Info ObjectiveFunc'    , 'alg': dqn_algorithm, 'suffix': 'disc_dqn_of'            , 'rewardF': informative_reward, 'init': None, 'noise': None},
	#	{'exe': True, 'desc': 'DQN with Info RewardFunc', 'alg': dqn_algorithm, 'suffix': 'disc2_dqn_rf_in',
	#	 'rewardF': informative_reward, 'init': initial_state, 'noise': initial_noise},
	#	{'exe': True, 'desc': 'DQN4 with Info RewardFunc', 'alg': dqn4_algorithm, 'suffix': 'disc4_dqn_rf_in',
	#	 'rewardF': informative_reward, 'init': initial_state, 'noise': initial_noise}
	#]

	# Load Parameters and Experiments from JSON File
	parameters = load_parameters(args.exp)

	global_params = parameters['global_params']

	model_dir = global_params['model_dir']
	plt_dir = global_params['plt_dir']
	data_dir = global_params['data_dir']

	show = global_params['show']

	# Run each experiment in the parameters file
	for experiment in parameters['experiments']:

		if experiment['exe']:

			# Experiment Parameters and objects
			desc = experiment['desc']
			runs = experiment['runs']
			episodes = experiment['episodes']
			time_steps = experiment['time_steps']
			test_episodes = experiment['test_episodes']

			init_state = experiment['initial_state']
			init_noise = experiment['initial_noise']

			reward_fun = experiment['reward_function']

			agent = experiment['agent']

			file_name = experiment['file_name']


			print_header(1, desc)

			run_train_stats = []
			run_test_stats = []

			for run in range(runs):

				print_header(2, 'RUN {}'.format(run + 1))
				print_header(3, 'Training')

				# Training
				env = ContinuousCartPoleEnv(reward_function=reward_fun)

				state_dim = env.observation_space.shape[0]
				action_dim = env.action_space.shape[0]


				# Clear weights
				agent.reset_parameters()

				# Train agent...
				stats = agent.train(env, episodes, time_steps, initial_state=init_state, initial_noise=init_noise)
				# ... and append statistics to list
				run_train_stats.append(stats)

				# Save agent checkpoint
				with open('{}/model_{}_run_{}_{}.pkl'.format(model_dir, file_name, run, timestamp()), 'wb') as f:
					pickle.dump(agent, f)

				# Run (deterministic) tests on the trained agent and save the statistics
				test_stats = test_agent(env, agent, episodes=test_episodes, time_steps=time_steps,
										initial_state=init_state, initial_noise=init_noise, render=show)
				run_test_stats.append(test_stats)


			# Concatenate stats for all runs ...
			train_rewards = []
			train_lengths = []
			train_losses  = []
			test_rewards  = []
			test_lengths  = []

			for r in range(runs):
				train_rewards.append(run_train_stats[r].episode_rewards)
				train_lengths.append(run_train_stats[r].episode_lengths)
				train_losses.append(run_train_stats[r].episode_loss)
				test_rewards.append(run_test_stats[r].episode_rewards)
				test_lengths.append(run_test_stats[r].episode_lengths)

			train_rewards = np.array(train_rewards)
			train_lengths = np.array(train_lengths)
			train_losses  = np.array(train_losses)
			test_rewards  = np.array(test_rewards)
			test_lengths  = np.array(test_lengths)

			# ... and store them in a dictionary
			plot_stats=[{'run': 'train', 'stats': {'rewards': train_rewards, 'lengths': train_lengths, 'losses': train_losses}},
						{'run': 'test' , 'stats': {'rewards': test_rewards , 'lengths': test_lengths}}]

			# Save Statistics
			with open(data_dir + '/data' + file_name + '_' + timestamp() + '.pkl', 'wb') as f:
				pickle.dump(plot_stats, f)

			# Plot Statistics
			plot_run_stats(plot_stats, dir=plt_dir, experiment=file_name, show=show)