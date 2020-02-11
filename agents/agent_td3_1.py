import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import torch.nn.functional as F

from td3 import TD3, MlpActor, MlpCritic
from action_functions import act_clipcont
from reward_functions import rf_spar_pos, rf_info_pos, rf_info2d_pos
from init import initial_states, initial_noises
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
file_name = 'ag_td3_1'
desc = 'TD3 Agent'

# Initial Conditions
init_state = None
init_noise = None


# Reward Function
reward_fun = rf_info2d_pos

# Actor
act_hid_non_linear = F.relu
act_hid_layers = 1
act_hid_dim = 20
act_out_non_linear = None
act_noise = 0.2
act_noise_clip = 0.5
actor = MlpActor(state_dim, action_dim, act_min=act_min, act_max=act_max, non_linearity=act_hid_non_linear,
				 hidden_layers=act_hid_layers, hidden_dim=act_hid_dim, output_non_linearity=act_out_non_linear,
				 noise=act_noise, noise_clip=act_noise_clip)

# Critic
crit_hid_non_linear = F.relu
crit_hid_layers = 1
crit_hid_dim = 20
crit_out_non_linear = None
critic = MlpCritic(state_dim, action_dim, non_linearity=crit_hid_non_linear, hidden_layers=crit_hid_layers,
				   hidden_dim=crit_hid_dim, output_non_linearity=crit_out_non_linear)

# Agent
lr = 1e-4
gamma = 0.99
tau = 0.01
policy_freq = 2
rb_max_size = 1e6
rb_batch_size = 64

agent = TD3(actor, critic, reward_fun, gamma=gamma, tau=tau, policy_freq=policy_freq, max_buffer_size=rb_max_size,
			batch_size=rb_batch_size, lr=lr)

# Training
show = False

train_agent(agent, desc, file_name, runs, episodes, time_steps, test_episodes, init_state, init_noise, show=show)
