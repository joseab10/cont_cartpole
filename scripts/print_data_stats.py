import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import numpy as np
import pickle

import argparse

from utils import print_header


def_dir = '../save/plots'
def_exp = 'plot_stats'

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store', default='', help='Path to the data file.', type=str)

args = parser.parse_args()

with open(args.file, 'rb') as f:
	stats = pickle.load(f)

print_header(1, 'Stats from file {}'.format(args.file))

for stat in stats:
	run = stat['run']
	run_stats = stat['stats']

	runs = run_stats['rewards'].shape[0]
	episodes = run_stats['rewards'].shape[1]

	print_header(2, '{}: {} runs x {} episodes'.format(run.title(), runs, episodes))

	avg_rw = np.mean(run_stats['rewards'], axis=0)
	sdv_rw = np.std(run_stats['rewards'], axis=0)
	avg_len = np.mean(run_stats['lengths'], axis=0)
	sdv_len = np.std(run_stats['lengths'], axis=0)

	for agent in range(runs):
		print_header(3, 'Run {}/{}:'.format(agent, runs))
		print('Rewards: mean = {}, stddev = {}'.format(agent, avg_rw[agent], sdv_rw[agent]))
		print('Lengths: mean = {}, stddev = {}'.format(agent, avg_len[agent], sdv_len[agent]))

	avg_rw = np.mean(run_stats['rewards'])
	sdv_rw = np.std(run_stats['rewards'])
	avg_len = np.mean(run_stats['lengths'])
	sdv_len = np.std(run_stats['lengths'])

	print_header(2, 'Global Stats')
	print('Rewards: mean = {}, stddev = {}'.format(avg_rw, sdv_rw))
	print('Lengths: mean = {}, stddev = {}'.format(avg_len, sdv_len))
