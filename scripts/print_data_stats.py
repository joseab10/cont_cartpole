import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import numpy as np
import pickle

import argparse

from utils import print_header, print_agg_stats


def_dir = '../save/plots'
def_exp = 'plot_stats'

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store', default='', help='Path to the data file.', type=str)

args = parser.parse_args()

with open(args.file, 'rb') as f:
	stats = pickle.load(f)

print_header(1, 'Stats from file {}'.format(args.file))

print_agg_stats(stats)
