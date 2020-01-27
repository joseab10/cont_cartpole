import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pandas as pd

from collections import namedtuple



def tt(ndarray, cuda=False):
	if cuda:
		return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
	else:
		return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])



def plot_episode_stats(stats, suffix='', smoothing_window=10, noshow=False):
	# Plot the episode length over time
	fig1 = plt.figure(figsize=(10, 5))
	plt.plot(stats.episode_lengths)
	plt.xlabel("Episode")
	plt.ylabel("Episode Length")
	plt.title("Episode Length over Time")
	fig1.savefig('episode_lengths' + suffix + '.png')
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
	fig2.savefig('reward' + suffix + '.png')
	if noshow:
		plt.close(fig2)
	else:
		plt.show(fig2)