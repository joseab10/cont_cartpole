import torch
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from collections import namedtuple

from datetime import datetime

from pathlib import Path


def mkdir(dir):
	if not Path(dir).exists():
		Path(dir).mkdir(parents=True, exist_ok=True)


def tt(ndarray):
	'''
	Converts an array to a Pytorch Tensor

	:param ndarray: (list|ndarray) The array to be converted
	:param cuda:    (bool) Use Cuda
	:return: 		(torch.Tensor) The converted array
	'''

	if not isinstance(ndarray, torch.Tensor):

		if not isinstance(ndarray, np.ndarray):
			ndarray = np.array(ndarray)

		if torch.cuda.is_available():
			ndarray = Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
		else:
			ndarray = Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

	return ndarray


def tn(value):
	'''
	Converts a value to a numpy ndarray

	:param value: () Value to be converted to a numpy array
	:return: 		(np.ndarray) Value as numpy ndarray
	'''

	if not isinstance(value, np.ndarray):

		if isinstance(value, torch.Tensor):
			value = value.detach().numpy()

		else:
			value = np.array(value)

	return value


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_loss"])


def print_header(level, msg):

	underline = ['', '#', '=', '-', '', '', '']

	print(''.join(['\n' for i in range(3 - level)]))
	print(''.join(['*' for i in range(4 - level)]) + ' ' + msg)
	print(''.join([underline[level] for i in range(80)]))


def print_stats(stats):

	msg = "{0[steps]:>6d} Steps, in Episode {0[episode]:>6d}/{0[episodes]:<6d}, Reward {0[reward]:=10.3f}"

	if 'loss' in stats:
		msg += ", Loss {0[loss]:=10.4f}"

	print(msg.format(stats))


def plot_stats(values, dir='', experiment='', run_type='', x_varname='', plot_agg=True, plot_runs=True, smoothing_window=10,
			   show=True, save=True):

	if experiment is not None or experiment != '':
		experiment = '_' + experiment

	if dir != '' and dir[-1] != '/':
		dir = dir + '/'

	fig = plt.figure(figsize=(10, 5))

	x_values = np.arange(1, values.shape[1] + 1)

	if plot_agg:
		means = np.mean(values, axis=0)
		stdev = np.std(values, axis=0)
		mins  = np.min(values, axis=0)
		maxs  = np.max(values, axis=0)

		medians = np.percentile(values, 50, axis=0)

		# Plot Extreme Area
		plt.fill_between(x_values, mins, maxs, alpha=0.125, label='Extremes')
		# Plot Mean +- 1*Sigma Area
		#plt.fill_between(x_values, means - stdev, means + stdev, alpha=0.25, label='1×σ')
		# Plot IQR Area
		plt.fill_between(x_values, np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0), alpha=0.35, label='IQR')
		# Plot Mean Curve
		#plt.plot(x_values, means, '--', label='Mean')
		# Plot Median Curve
		plt.plot(x_values, medians, '--', label='Median', linewidth=1.5)

	# Plot individual runs
	if plot_runs:
		for i in range(values.shape[0]):
			if smoothing_window > 3 * values.shape[1]:
				values = pd.Series(values[i, :]).rolling(smoothing_window, min_periods=smoothing_window).mean()

			plt.plot(x_values, values[i,:], label='Run {}'.format(i + 1), linewidth=0.25)

	# Plot Information
	plt.xlabel("Episode")
	plt.ylabel("Episode " + x_varname)
	plt.title(run_type + "Episode " + x_varname + " over Time")
	plt.legend()

	# Save Plot as png
	if save:
		mkdir(dir)
		fig.savefig('{}plot{}_ep_{}_{}.png'.format(dir, experiment, x_varname.lower() + 's', timestamp()))

	if show:
		plt.show(fig)
	else:
		plt.close(fig)


def plot_run_stats(stats, dir='', experiment='', plot_runs=True, plot_agg=True, smoothing_window=10, show=True, save=True):

	for runs in stats:
		run = runs['run']
		substats = runs['stats']

		for varname, substat in substats.items():

			if not np.all(substat == 0):

				plot_stats(substat, dir=dir + '/' + experiment, experiment=experiment + '_' + run, run_type=run.title() + ' ',
						   x_varname=varname.title(), plot_runs=plot_runs, plot_agg=plot_agg,
						   smoothing_window=smoothing_window, show=show, save=save)


def timestamp():
	return datetime.now().strftime("%Y%m%d_%H%M%S")