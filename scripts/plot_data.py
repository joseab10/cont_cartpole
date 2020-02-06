import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import pickle

from utils import *
import argparse


def_dir = '../save/plots'
def_exp = 'plot_stats'

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store'     , default=''     , help='Path to the data file.'               , type=str)

parser.add_argument('--nosh', action='store_true', default=False   , help='Do not display the plots.'           )

parser.add_argument('--save', action='store_true', default=False  , help='Save the plots.'                      )
parser.add_argument('--dir' , action='store'     , default=def_dir, help='Path to the data file.'               , type=str)
parser.add_argument('--exp' , action='store'     , default=def_exp, help='Experiment name (used for filename).' , type=str)

parser.add_argument('--runs', action='store_true', default=False  , help='Plot individual runs.')
parser.add_argument('--nagg', action='store_true', default=False  , help='Do not plot aggregate stats')
parser.add_argument('--smw' , action='store'     , default=10     , help='Smoothing window'                    , type=int)

args = parser.parse_args()

with open(args.file, 'rb') as f:
	stats = pickle.load(f)

plot_run_stats(stats, path=args.dir, experiment=args.exp,
			   plot_runs=args.runs, plot_agg=not args.nagg, smth_wnd=args.smw,
			   show=not args.nosh, save=args.save)
