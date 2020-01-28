import torch
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from collections import namedtuple

from datetime import datetime



def tt(ndarray, cuda=False):
	if cuda:
		return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
	else:
		return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def tn(list):
	if not isinstance(list, np.ndarray):
		list = np.array(list)
	return list


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, dir='', experiment='', smoothing_window=10, noshow=False):

	if experiment is not None or experiment != '':
		experiment = '_'  + experiment

	if dir != '' and dir[-1] != '/':
		dir = dir + '/'

	# Plot the episode length over time
	fig1 = plt.figure(figsize=(10, 5))
	plt.plot(stats.episode_lengths)
	plt.xlabel("Episode")
	plt.ylabel("Episode Length")
	plt.title("Episode Length over Time")
	fig1.savefig('{}ep_lengths{}{}.png'.format(dir, experiment, timestamp()))
	if noshow:
		plt.close(fig1)
	else:
		plt.show(fig1)

	# Plot the episode reward over time
	fig2 = plt.figure(figsize=(10, 5))
	rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	plt.plot(rewards_smoothed)
	plt.xlabel("Episode")
	plt.ylabel("Episode Reward (Smoothed)")
	plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
	fig2.savefig('{}ep_rewards{}{}.png'.format(dir, experiment, timestamp()))
	if noshow:
		plt.close(fig2)
	else:
		plt.show(fig2)


def plot_mean_stdev(stats, dir='', experiment='', plot_runs=True, smoothing_window=10, noshow=False):

	if experiment is not None or experiment != '':
		experiment = '_'  + experiment

	if dir != '' and dir[-1] != '/':
		dir = dir + '/'

	# Plot the episode length over time
	fig1 = plt.figure(figsize=(10, 5))


	lengths = stats['lengths']
	lengths_means = stats['length_means']
	lengths_stdev = stats['length_stdev']

	x = np.arange(1, lengths.shape[1] + 1)

	plt.fill_between(x, np.min(lengths, axis=0), np.max(lengths, axis=0), alpha=0.125, label='Extremes')
	plt.fill_between(x, lengths_means - lengths_stdev, lengths_means + lengths_stdev, alpha=0.25, label='Stdev')
	plt.plot(x, lengths_means, '--', label='Mean')

	if plot_runs:
		for i in range(lengths.shape[0]):
			plt.plot(x, lengths[i,:], label='Run {}'.format(i + 1), linewidth=0.5)

	#plt.plot(stats['lengths'])
	plt.xlabel("Episode")
	plt.ylabel("Episode Length")
	plt.title("Episode Length over Time")
	plt.legend()
	fig1.savefig('{}plot{}_ep_lengths_{}.png'.format(dir, experiment, timestamp()))
	if noshow:
		plt.close(fig1)
	else:
		plt.show(fig1)


	# Plot the episode length over time
	fig2 = plt.figure(figsize=(10, 5))

	rewards = stats['rewards']
	rewards_means = stats['reward_means']
	rewards_stdev = stats['reward_stdev']

	x = np.arange(1, rewards.shape[1] + 1)

	plt.fill_between(x, np.min(rewards, axis=0), np.max(rewards, axis=0), alpha=0.125, label='Extremes')
	plt.fill_between(x, rewards_means - rewards_stdev, rewards_means + rewards_stdev, alpha=0.25, label='Stdev')
	plt.plot(x, rewards_means, '--', label='Mean')

	if plot_runs:
		for i in range(rewards.shape[0]):
			rewards_smoothed = pd.Series(rewards[i, :]).rolling(smoothing_window,
																		min_periods=smoothing_window).mean()
			plt.plot(x, rewards_smoothed, label='Run {}'.format(i + 1), linewidth=0.5)

	# plt.plot(stats['lengths'])
	plt.xlabel("Episode")
	plt.ylabel("Episode Reward")
	plt.title("Episode Reward over Time")
	plt.legend()
	fig2.savefig('{}plot{}_ep_reward_{}.png'.format(dir, experiment, timestamp()))
	if noshow:
		plt.close(fig2)
	else:
		plt.show(fig2)


def plot_run_stats(stats, dir='', experiment='', plot_runs=True, smoothing_window=10, noshow=False):

	for stat in stats:
		plot_mean_stdev(stat, dir=dir, experiment=experiment + '_' + stat['run'], plot_runs=plot_runs, smoothing_window=smoothing_window, noshow=noshow)


def timestamp():
	return datetime.now().strftime("%Y%m%d_%H%M%S")