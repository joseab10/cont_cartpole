import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_aggregate, mkdir, timestamp


def_dir = '../save/plots/compare/'
def_exp = 'plot_compare'

parser = argparse.ArgumentParser()

parser.add_argument('--files', action='store', help='Comma Separated List of Stat files to plot', type=str)
parser.add_argument('--labels', action='store', help='Comma Separated List of labels for each curve', type=str)

parser.add_argument('--nosh', action='store_true', default=False   , help='Do not display the plots.'           )

parser.add_argument('--save', action='store_true', default=False  , help='Save the plots.'                      )
parser.add_argument('--dir' , action='store'     , default=def_dir, help='Path to the plot file.' , type=str)
parser.add_argument('--fname' , action='store'     , default=def_exp, help='Filename for the plots.' , type=str)


parser.add_argument('--smw' , action='store'     , default=50     , help='Smoothing window' , type=int)

args = parser.parse_args()

files = args.files.split(',')
labels = args.labels.split(',')

if len(files) != len(labels):
	raise IndexError('The number of files and labels does not match')

stats = {}
runs_dict = {}

# Restructure stats into {run1:{var_1:{label_1: stats_1, ..., label_n: stats_,n}, ..., var_m:{...}}, ..., {run_o:{}}
for i, file in enumerate(files):
	with open(file, 'rb') as f:
		file_stats = pickle.load(f)

	for runs in file_stats:
		run = runs['run']
		substats = runs['stats']

		if run in runs_dict:
			variable_dic = runs_dict[run]
		else:
			variable_dic = {}

		for var_name, substat in substats.items():
			if not np.all(substat == 0):

				if var_name in variable_dic:
					variable_dic[var_name][labels[i]] = substat
				else:
					variable_dic[var_name] = {labels[i]: substat}

		if run not in runs_dict:
			runs_dict[run] = variable_dic

# Plot aggregate values for each run and variable
for run, variables in runs_dict.items():
	for var_name, stats in variables.items():

		fig = plt.figure(figsize=(10, 5))

		for label, fstats in stats.items():
			plot_aggregate(fstats, label=label, smth_wnd=args.smw, plot_mean=False, plot_stdev=False,
						   plot_med=True, plot_iqr=True, plot_ext=False)

		# Plot Information
		plt.xlabel("Episode")
		plt.ylabel("Episode " + var_name)
		plt.title("{} Episode {} over Time".format(run.title(), var_name))
		plt.legend()

		# Save Plot as png
		if args.save:
			mkdir(args.dir)
			fig.savefig(
				'{}plot_{}_{}_ep_{}_{}.png'.format(args.dir, args.fname, run.lower(), var_name.lower(), timestamp()))

		if args.nosh:
			plt.close(fig)
		else:
			plt.show(fig)
