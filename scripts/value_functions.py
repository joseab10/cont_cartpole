import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *



class StateValueFunction(nn.Module):
	def __init__(self, state_dim, non_linearity=F.relu, hidden_layers=1, hidden_dim=20):

		super(StateValueFunction, self).__init__()

		self._non_linearity = non_linearity
		# Variable list of Layers
		self.layers = nn.ModuleList()

		# Input Layer
		self.layers.append(nn.Linear(state_dim, hidden_dim))

		# Hiden Layers
		for i in range(hidden_layers):
			self.layers.append(nn.Linear(hidden_dim, hidden_dim))

		# Output Layers
		self.layers.append(nn.Linear(hidden_dim, 1))



	def forward(self, x):

		if not isinstance(x, torch.Tensor):
			x = tt(x)

		for i in range(len(self.layers) - 1):
			x = self._non_linearity(self.layers[i](x))

		x = self.layers[-1](x)

		return x