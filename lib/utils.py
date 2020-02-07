import torch
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from collections import namedtuple

from datetime import datetime

from pathlib import Path


def mkdir(path):
	""" Creates a directory if it doesn't already exist

	:param path: (str) Path to the directory to be created
	:return:    None
	"""
	if not Path(path).exists():
		Path(path).mkdir(parents=True, exist_ok=True)


def tt(ndarray):
	""" Converts an array to a Pytorch Tensor

	:param ndarray: (list, ndarray) The array to be converted
	:return: 		(torch.Tensor) The converted array
	"""

	if not isinstance(ndarray, torch.Tensor):

		if not isinstance(ndarray, np.ndarray):
			ndarray = np.array(ndarray)

		if torch.cuda.is_available():
			ndarray = Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
		else:
			ndarray = Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

	return ndarray


def tn(value):
	""" Converts a value to a numpy ndarray

	:param value: () Value to be converted to a numpy array
	:return: 	  (np.ndarray) Value as numpy ndarray
	"""

	if not isinstance(value, np.ndarray):

		if isinstance(value, torch.Tensor):
			value = value.detach()
			value = value.numpy()

		else:
			value = np.array(value)

	return value


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_loss"])


def print_header(level, msg):
	""" Prints a line to the otuput with preceding blank lines and some underline styles for easier interpretation of the
	results.

	:param level: (int) Heading level (1 - Highest,  3 - Lowest)
	:param msg:   (str) Message to be output
	:return:      None
	"""

	underline = ['', '#', '=', '-', '', '', '']

	print(''.join(['\n' for _ in range(3 - level)]))
	print(''.join(['*' for _ in range(4 - level)]) + ' ' + msg)
	print(''.join([underline[level] for _ in range(80)]))


def print_stats(stats, extra_str=''):
	""" Prints the Training/Testing statistics in a preformated way for easier reading

	stats is a dictionary that must include:
		· "steps"   : (int)   the number of steps the episode took
		· "episode" : (int)   the number of episode in a training/testing session
		· "episodes": (int)   the total number of episodes to be trained/tested
		· "reward"  : (float) the accumulated rewards obtained by the agent during the episode
		· "loss"    : (float) [Optional] The average or accumulated loss function from an optimization problem

	:param stats:     (dict) Dictionary containing the aforementioned statistics
	:param extra_str: (str)  String to be appended at the end of the line
	:return:          None
	"""

	msg = "{0[steps]:>6d} Steps, in Episode {0[episode]:>6d}/{0[episodes]:<6d}, Reward {0[reward]:=10.3f}"

	if 'loss' in stats:
		msg += ", Loss {0[loss]:=10.4f}"

	print(msg.format(stats), extra_str)


def plot_stats(values, path='', experiment='', run_type='', x_var_name='', plot_agg=True, plot_runs=True, smth_wnd=10,
			   show=True, save=True):
	""" Plots the statistics for a single variable over multiple runs.

	If save=True, then the file will be saved with the following filename:

		<dir>/plot_<experiment>_<run_type>_ep_<x_varname>_<timestamp>.png

	:param values:     (ndarray) 1D or 2D array containing the stats for a single variable (e.g. rewards) over
	                               one or more runs
	:param path:       (str)     Directory Path for saving the resulting plot
	:param experiment: (str)     Common part of the filename
	:param run_type:   (str)     Type of session run (e.g. "training", "testing") for both the plot title and filename
	:param x_var_name: (str)     Name of the variable being plotted (e.g. "reward", "length") for the plot title,
	                               axis and filename
	:param plot_agg:   (bool)    Plot aggregate information (extremes, median and IQR) for each episode over all runs
	:param plot_runs:  (bool)    Plot a curve for each individual run
	:param smth_wnd:   (int)     Running average window for smoothing the noisy individual run curves
	:param show:       (bool)    Show plot during execution
	:param save:       (bool)    Save plot to a file with filename constructed as in the description
	:return:           None
	"""

	if experiment is not None or experiment != '':
		experiment = '_' + experiment

	if path != '' and path[-1] != '/':
		path = path + '/'

	fig = plt.figure(figsize=(10, 5))

	x_values = np.arange(1, values.shape[1] + 1)

	smoothen = True if 0 < 3 * smth_wnd < values.shape[1] else False

	if plot_agg:
		# means = np.mean(values, axis=0)
		# std_dev = np.std(values, axis=0)
		medians = np.percentile(values, 50, axis=0)

		ext_min  = np.min(values, axis=0)
		ext_max  = np.max(values, axis=0)

		iqr_25 = np.percentile(values, 25, axis=0)
		iqr_75 = np.percentile(values, 75, axis=0)

		if smoothen:
			medians = pd.Series(medians).rolling(smth_wnd, min_periods=smth_wnd).mean()
			ext_min = pd.Series(ext_min).rolling(smth_wnd, min_periods=smth_wnd).mean()
			ext_max = pd.Series(ext_max).rolling(smth_wnd, min_periods=smth_wnd).mean()
			iqr_25  = pd.Series(iqr_25).rolling(smth_wnd, min_periods=smth_wnd).mean()
			iqr_75  = pd.Series(iqr_75).rolling(smth_wnd, min_periods=smth_wnd).mean()

		# Plot Extreme Area
		plt.fill_between(x_values, ext_min, ext_max, alpha=0.125, label='Extremes')
		# Plot Mean +- 1*Sigma Area
		# plt.fill_between(x_values, means - std_dev, means + std_dev, alpha=0.25, label='1×σ')
		# Plot IQR Area
		plt.fill_between(x_values, iqr_25, iqr_75, alpha=0.45, label='IQR')
		# Plot Mean Curve
		# plt.plot(x_values, means, '--', label='Mean')
		# Plot Median Curve
		plt.plot(x_values, medians, '--', label='Median', linewidth=1.5)

	# Plot individual runs
	if plot_runs:
		for i in range(values.shape[0]):

			if len(values.shape) == 1:
				run_values = values[i]
			else:
				run_values = values[i, :]

			if smoothen:
				run_values = pd.Series(run_values).rolling(smth_wnd, min_periods=smth_wnd).mean()

			plt.plot(x_values, run_values, label='Run {}'.format(i + 1), linewidth=0.25)

	# Plot Information
	plt.xlabel("Episode")
	plt.ylabel("Episode " + x_var_name)
	plt.title("{} Episode {} over Time".format(run_type.title(), x_var_name))
	plt.legend()

	# Save Plot as png
	if save:
		mkdir(path)
		fig.savefig('{}plot_{}_{}_ep_{}_{}.png'.format(path, experiment, run_type.lower(), x_var_name.lower(), timestamp()))

	if show:
		plt.show(fig)
	else:
		plt.close(fig)


def plot_run_stats(stats, path='', experiment='', plot_runs=True, plot_agg=True, smth_wnd=10,
				   show=True, save=True):
	""" Plots all of the statistics from a dictionary into individual plots.

	The dictionary must have the structure:

	stats = [
		{'run': <run_type>, 'stats': {<x1_name>: <x1_val>, <x2_name>: <x2_val>, ...},
	]

	where:
		· <run_type>:  (str)     can be for example "train", "test"
		· <xn_name> :  (str)     name of the variable to be plotted, e.g.: "rewards" or "lengths"
		· <xn_val>  :  (ndarray) 1D or 2D array of values. If 2D, dim 0 is for episodic data, while dim 1 is for each run

	If save=True, then the files will be saved with the following filenames:

		<dir>/plot_<experiment>_<run_type>_ep_<x_varname>_<timestamp>.png


	:param stats:      (list) List of dictionaries containing the statistics. Structure in the description.
	:param path:       (str)  Directory Path for saving the resulting plot
	:param experiment: (str)  Common part of the filename for all runs and variables
	:param plot_runs:  (bool) Plot aggregate information (extremes, median and IQR) for each episode over all runs
	:param plot_agg:   (bool) Plot a curve for each individual run
	:param smth_wnd:   (int)  Running average window for smoothing the noisy individual run curves
	:param show:       (bool) Show plot during execution
	:param save:       (bool) Save plot to a file with filename constructed as in the description
	:return:           None
	"""
	for runs in stats:
		run = runs['run']
		substats = runs['stats']

		for varname, substat in substats.items():

			if not np.all(substat == 0):

				plot_stats(substat, path=path + '/' + experiment, experiment=experiment, run_type=run,
						   x_var_name=varname.title(), plot_runs=plot_runs, plot_agg=plot_agg,
						   smth_wnd=smth_wnd, show=show, save=save)


def timestamp():
	""" Returns the current time. Used for file names mostly

	:return: (str) Current time formatted as YYmmdd_HHMMSS
	"""
	return datetime.now().strftime("%Y%m%d_%H%M%S")
