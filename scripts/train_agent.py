import numpy as np
import pickle

from continuous_cartpole import ContinuousCartPoleEnv
from utils import mkdir, timestamp, EpisodeStats, print_header, print_stats, plot_run_stats, tn


def test_agent(env, agent, episodes=5, time_steps=500, initial_state=None, initial_noise=None, render=True):

	stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes),
						 episode_loss=np.zeros(episodes))

	print_header(3, 'Testing')

	for e in range(episodes):

		s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)

		for t in range(time_steps):

			if render:
				env.render()

			a = agent.get_action(s, deterministic=True)
			s, r, d, _ = env.step(tn(a))

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


def train_agent(agent, desc='Agent1', file_name='agent1', runs=5, episodes=5000, time_steps=300, test_episodes=10,
				init_state=None, init_noise=None, reward_fun=None,
				model_dir='../save/models', data_dir='../save/stats', plt_dir='../save/plots',
				show=False):

	print_header(1, desc)

	run_train_stats = []
	run_test_stats = []

	for run in range(runs):
		print_header(2, 'RUN {}'.format(run + 1))
		print_header(3, 'Training')

		# Training
		env = ContinuousCartPoleEnv(reward_function=reward_fun)

		# Clear weights
		agent.reset_parameters()

		# Train agent...
		stats = agent.train(env, episodes, time_steps, initial_state=init_state, initial_noise=init_noise)
		# ... and append statistics to list
		run_train_stats.append(stats)

		# Save agent checkpoint
		exp_model_dir = model_dir + '/' + file_name
		mkdir(exp_model_dir)
		with open('{}/model_{}_run_{}_{}.pkl'.format(exp_model_dir, file_name, run + 1, timestamp()),
					'wb') as f:
			pickle.dump(agent, f)

		# Run (deterministic) tests on the trained agent and save the statistics
		test_stats = test_agent(env, agent, episodes=test_episodes, time_steps=time_steps,
								initial_state=init_state, initial_noise=init_noise, render=show)
		run_test_stats.append(test_stats)

	# Concatenate stats for all runs ...
	train_rewards = []
	train_lengths = []
	train_losses = []
	test_rewards = []
	test_lengths = []

	for r in range(runs):
		train_rewards.append(run_train_stats[r].episode_rewards)
		train_lengths.append(run_train_stats[r].episode_lengths)
		train_losses.append(run_train_stats[r].episode_loss)
		test_rewards.append(run_test_stats[r].episode_rewards)
		test_lengths.append(run_test_stats[r].episode_lengths)

	train_rewards = np.array(train_rewards)
	train_lengths = np.array(train_lengths)
	train_losses = np.array(train_losses)
	test_rewards = np.array(test_rewards)
	test_lengths = np.array(test_lengths)

	# ... and store them in a dictionary
	plot_stats = [
		{'run': 'train', 'stats': {'rewards': train_rewards, 'lengths': train_lengths, 'losses': train_losses}},
		{'run': 'test', 'stats': {'rewards': test_rewards, 'lengths': test_lengths}}]

	# Save Statistics
	exp_stats_dir = data_dir + '/' + file_name
	mkdir(exp_stats_dir)
	with open('{}/stats_{}_{}.pkl'.format(exp_stats_dir, file_name, timestamp()), 'wb') as f:
		pickle.dump(plot_stats, f)

	# Plot Statistics
	plot_run_stats(plot_stats, path=plt_dir, experiment=file_name, show=show)


def run_experiments(experiments):

	for experiment in experiments['experiments']:

		if experiment['exe']:

			# Remove the 'exe' key from the dictionary and pass all the rest as arguments to the train_agent function
			xp = {x: experiment[x] for x in experiment if x not in ['exe']}

			train_agent(**xp)
