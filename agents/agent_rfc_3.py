import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import torch.nn.functional as F

from policies import GaussPolicy, BetaPolicy, MlpPolicy
from continuous_reinforce import REINFORCE
from action_functions import act_clipcont
from reward_functions import rf_spar_pos, rf_info_pos, rf_info2d_pos, rf_info2d_sharp_pos
from init import initial_states, initial_noises
from value_functions import StateValueFunction
from train import train_agent


runs = 5
episodes = 5000
time_steps = 300
test_episodes = 10

state_dim = 4
action_dim = 1

act_min = -1
act_max = 1

# File and output names
file_name = 'ag_rfc_3'
desc = 'Continuous Reinforce'

# Initial Conditions
init_state = None
init_noise = initial_noises['360']

# Action Function
act_fun = act_clipcont

# Reward Function
reward_fun = rf_info2d_pos

# Policy
a_non_linear = F.relu
a_hid_layers = 1
a_hid_dim    = 20
b_non_linear = F.relu
b_hid_layers = 1
b_hid_dim    = 20
ns = True  # Numerically Stable

policy = BetaPolicy(state_dim, action_dim=action_dim, act_min=act_min, act_max=act_max,
					input_non_linearity=a_non_linear, input_hidden_layers=a_hid_layers, input_hidden_dim=a_hid_dim,
					ns=ns)

# Baseline Function
bl = True

bl_lr = 1e-4

bl_non_linear = F.relu
bl_hid_layers = 1
bl_hid_dim    = 20

bl_fun = StateValueFunction(state_dim, non_linearity=bl_non_linear, hidden_layers=bl_hid_layers, hidden_dim=bl_hid_dim)


# Agent
lr = 1e-4
gamma = 0.99

agent = REINFORCE(policy, act_clipcont, state_dim, action_dim, gamma=gamma, baseline=bl, baseline_fun=bl_fun,
				  reward_fun=reward_fun, lr=lr, bl_lr=bl_lr)

# Training
show = False

train_agent(agent, desc, file_name, runs, episodes, time_steps, test_episodes, init_state, init_noise, show=show)
