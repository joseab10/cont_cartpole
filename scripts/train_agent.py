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

	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', action='store', default='experiments_example.json', help='Path to the experiments file.', type=str)

	args = parser.parse_args()

	# Load Parameters and Experiments from JSON File
	parameters = load_parameters(args.exp)

	global_params = parameters['global_params']

	model_dir = global_params['model_dir']
	plt_dir = global_params['plt_dir']
	data_dir = global_params['data_dir']

	mkdir(model_dir)
	mkdir(plt_dir)
	mkdir(data_dir)

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
				exp_model_dir = model_dir + '/' + file_name
				mkdir(exp_model_dir)
				with open('{}/model_{}_run_{}_{}.pkl'.format(exp_model_dir, file_name, run + 1, timestamp()), 'wb') as f:
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
			exp_stats_dir = data_dir + '/' + file_name
			mkdir(exp_stats_dir)
			with open('{}/stats_{}_{}.pkl'.format(exp_stats_dir, file_name, timestamp()), 'wb') as f:
				pickle.dump(plot_stats, f)

			# Plot Statistics
			plot_run_stats(plot_stats, dir=plt_dir, experiment=file_name, show=show)