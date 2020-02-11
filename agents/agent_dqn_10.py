import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import torch.nn.functional as F

from continuous_reinforce import REINFORCE
from deep_q_learning import DQN
from policies import GaussPolicy, BetaPolicy, MlpPolicy, EpsilonGreedyPolicy
from value_functions import StateValueFunction, StateActionValueFunction
from action_functions import act_disc2cont
from reward_functions import rf_spar_pos, rf_info_pos, rf_info2d_pos, rf_info2d_sharp_pos
from schedules import sch_exp_decay, sch_linear_decay, sch_no_decay, Schedule
from init import initial_states, initial_noises
from continuous_cartpole import ContinuousCartPoleEnv
from train import train_agent


runs = 5
episodes = 5000
time_steps = 300
test_episodes = 10

state_dim = 4
action_dim = 2

# File and output names
file_name = 'ag_dqn_10'
desc = 'DDQN'

# Initial Conditions
init_state = None
init_noise = initial_noises['360']

# Value Functions
q_non_linear    = F.relu
q_hidden_layers = 1
q_hidden_dim    = 20

Q = StateActionValueFunction(state_dim, action_dim, non_linearity=q_non_linear, hidden_layers=q_hidden_layers,
								 hidden_dim=q_hidden_dim)
Q_target = StateActionValueFunction(state_dim, action_dim, non_linearity=q_non_linear, hidden_layers=q_hidden_layers,
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

# Reward Function
reward_fun = rf_info2d_sharp_pos

# Action Pre/Post-Processing Action
act_fun = act_disc2cont

# Agent
lr = 1e-4
gamma = 0.98
doubleQ = True # Run doubleQ-DQN sampling from Q_target and bootstraping from Q
rb = True
rb_max_size = 1e6
rb_batch_size = 64
tau = 0.01

agent = DQN(policy, act_fun, Q, Q_target, state_dim, action_dim, gamma, doubleQ, reward_fun=reward_fun,
			replay_buffer=rb, max_buffer_size=rb_max_size, batch_size=rb_batch_size, tau=tau, lr=lr)

# Training
show = False

train_agent(agent, desc, file_name, runs, episodes, time_steps, test_episodes, init_state, init_noise, show=show)
