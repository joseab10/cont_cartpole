import sys

from continuous_reinforce import REINFORCE
from policies import BetaPolicy, GaussPolicy
from value_functions import StateValueFunction, StateActionValueFunction
from neural_networks import MLP
from action_functions import act_clipcont

from torch.utils.tensorboard import SummaryWriter

import torch

# Added to be able to find python files outside of cwd
sys.path.append('../lib')

state_dim = 4
action_dim = 1

s = torch.randn(state_dim)
a = torch.randn(action_dim)

nn = MLP(state_dim, action_dim)

pib = BetaPolicy(state_dim, action_dim, -1, 1)
pig = GaussPolicy(state_dim, action_dim)
baseline = StateValueFunction(state_dim)
agent = REINFORCE(pib, act_clipcont, state_dim, action_dim, 0.99, baseline=True, baseline_fun=baseline)


Q = StateActionValueFunction(state_dim, 4)
Qt = StateActionValueFunction(state_dim, 4)

writer = SummaryWriter()
#writer.add_graph(pib, (s, a))
#writer.add_graph(pig, (s,a))
writer.add_graph(Q, s)
#writer.add_graph(Qt, s)
writer.close()