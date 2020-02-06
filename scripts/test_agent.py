import sys

import pickle

from continuous_cartpole import ContinuousCartPoleEnv

from train import test_agent, plot_run_stats

import argparse

import reward_functions
from init import initial_states, initial_noises

# Added to be able to find python files outside of cwd
sys.path.append('../lib')

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store'  , default=''    , help='Path to the model file.', type=str)

parser.add_argument('--ep'    , action='store', default=5     , help='Number of test episodes.', type=int)
parser.add_argument('--ts'    , action='store', default=500   , help='Time steps per episode.', type=int)
parser.add_argument('--inist' , action='store', default='none', help='Initial State.', type=str)
parser.add_argument('--inirnd', action='store', default='none', help='Initial Noise.', type=str)
parser.add_argument('--rew'   , action='store', default='none', help='Reward Function.', type=str)

parser.add_argument('--smw'   , action='store', default=10    , help='Smoothing window', type=int)

args = parser.parse_args()

reward_function = getattr(reward_functions, args.rew)
initial_state = initial_states[args.inist]
initial_noise = initial_noises[args.inirnd]

env = ContinuousCartPoleEnv(reward_function=reward_function)

with open(args.file, 'rb') as f:
	agent = pickle.load(f)

stats = test_agent(env, agent, episodes=args.ep, time_steps=args.ts, initial_state=initial_state,
				   initial_noise=initial_noise, render=True)

plt_stats = [{'run': 'test', 'stats': {'rewards': stats.episode_rewards.reshape([1, args.ep]),
										'lengths': stats.episode_lengths.reshape([1, args.ep])}}]

plot_run_stats(plt_stats, plot_runs=True, plot_agg=False, smth_wnd=args.smw,
			   show=True, save=False)
