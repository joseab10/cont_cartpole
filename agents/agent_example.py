import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import torch.nn.functional as F

from deep_q_learning import DQN
from policies import EpsilonGreedyPolicy
from value_functions import StateActionValueFunction
from action_functions import act_disc2cont
from reward_functions import rf_info_pos
from continuous_cartpole import ContinuousCartPoleEnv
from train import train_agent
from schedules import sch_exp_decay, sch_linear_decay, sch_no_decay, Schedule


file_name = 'ag_dqn_nobuff'

show = False

desc = 'DQN without Replay Buffer'
runs = 2
episodes = 100
time_steps = 300
test_episodes = 10

state_dim = 4
action_dim = 2

q_hidden_layers = 1
q_hidden_dim    = 20

epsilon = 0.2

gamma = 0.99
doubleQ = True # Run doubleQ-DQN sampling from Q_target and bootstraping from Q

lr = 1e-4

act_fun = act_disc2cont
reward_fun = rf_info_pos

init_state = None
init_noise = None

# Value Functions
Q = StateActionValueFunction(state_dim, action_dim, non_linearity=F.relu, hidden_layers=q_hidden_layers,
								 hidden_dim=q_hidden_dim)
Q_target = StateActionValueFunction(state_dim, action_dim, non_linearity=F.relu, hidden_layers=q_hidden_layers,
									hidden_dim=q_hidden_dim)

# Epsilon Schedule
t0 = 100
t1 = 1200
e0 = 0.90
e1 = 0.10
decay_fun = sch_exp_decay
cos_ann = True
ann_cyc = 5
schedule = Schedule(t0, t1, e0, e1, decay_fun, cosine_annealing=cos_ann, annealing_cycles=ann_cyc)

# Policy
policy = EpsilonGreedyPolicy(schedule=schedule, value_function=Q)

# Agent
rb = False
agent = DQN(policy, act_fun, Q, Q_target, state_dim, action_dim, gamma, doubleQ, reward_fun=reward_fun,
			replay_buffer=rb, lr=lr)

# Environment
env = ContinuousCartPoleEnv(reward_function=reward_fun)

train_agent(agent, desc, file_name, runs, episodes, time_steps, test_episodes, init_state, init_noise, show=show)