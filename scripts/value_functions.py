from neural_networks import *


class StateActionValueFunction(MLP):
	def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_layers=1, hidden_dim=20):

		super(StateActionValueFunction, self).__init__(state_dim, action_dim,hidden_dim=hidden_dim,
													   hidden_non_linearity=non_linearity, hidden_layers=hidden_layers)


class StateValueFunction(StateActionValueFunction):

	def __init__(self, state_dim, non_linearity=F.relu, hidden_layers=1, hidden_dim=20):

		super(StateValueFunction, self).__init__(state_dim, 1, non_linearity=non_linearity,
												 hidden_layers=hidden_layers, hidden_dim=hidden_dim)