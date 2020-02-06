import sys
# Added to be able to find python files outside of cwd
sys.path.append('../lib')

import torch
from torch.utils.tensorboard import SummaryWriter

from continuous_reinforce import REINFORCE
from policies import BetaPolicy, GaussPolicy
from value_functions import StateValueFunction, StateActionValueFunction
from neural_networks import MLP
from action_functions import act_clipcont


tb_dir = '../save/tb_graphs'

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

graphs = [
	{'model': pib, 'inputs': (s, a)},
	{'model': pig, 'inputs': (s, a)},
	{'model': Q  , 'inputs': s},
	{'model': Qt , 'inputs': s}
]

for graph in graphs:

	writer = SummaryWriter(log_dir=tb_dir)
	writer.add_graph(Q, s)
	writer.close()
