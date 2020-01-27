import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import *


class GaussPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, mean_non_linearity=F.relu, mean_hidden_layers=1, mean_hidden_dim=20,
											   var_non_linearity=F.relu,  var_hidden_layers=1,  var_hidden_dim=20):

		super(GaussPolicy, self).__init__()

		self._action_dim = action_dim

		self._mean_non_linearity = mean_non_linearity
		self._var_non_linearity  = var_non_linearity

		# Variable list of Layers
		self.mean_layers = nn.ModuleList()
		self.var_layers  = nn.ModuleList()

		# Input Layers for Mean and Variance
		self.mean_layers.append(nn.Linear(state_dim, mean_hidden_dim))
		self.var_layers.append(nn.Linear(state_dim, var_hidden_dim))

		# Hidden Layers for Mean
		for i in range(mean_hidden_layers):
			self.mean_layers.append(nn.Linear(mean_hidden_dim, mean_hidden_dim))

		# Hidden Layers for Variance
		for i in range(var_hidden_layers):
			self.var_layers.append(nn.Linear(var_hidden_dim, var_hidden_dim))

		# Output Layers for Mean and Variance
		self.mean_layers.append(nn.Linear(mean_hidden_dim, action_dim))
		self.var_layers.append(nn.Linear(var_hidden_dim, action_dim))


	def forward(self, x, a):

		if not isinstance(x, torch.Tensor):
			x = tt(x)

		if not isinstance(a, torch.Tensor):
			a = tt(a)

		mu = x
		sigma = x

		# Input and Hidden Layers for Mean
		for i in range(len(self.mean_layers) - 1):
			mu = self._mean_non_linearity(self.mean_layers[i](mu))

		# Input and Hidden Layers for Variance
		for i in range(len(self.var_layers) - 1):
			sigma = self._var_non_linearity(self.var_layers[i](sigma))

		mu = self.mean_layers[-1](mu)
		sigma = F.softplus(self.var_layers[-1](sigma)) #sigma must always be > 0

		self.mu    = mu
		self.sigma = sigma

		sigma2 = sigma**2

		p = 1 / torch.sqrt(2 * np.pi * sigma2) * torch.exp(torch.div(- (mu - a)**2, 2 * sigma2))

		return p

	def get_action(self, s):

		if not hasattr(self, 'mu') or not hasattr(self, 'sigma'):
			self.forward(s, torch.rand(self._action_dim))



		m = torch.distributions.normal.Normal(self.mu, self.sigma)
		a = m.sample()

		if self._action_dim > 1:
			p = self.forward(s, a)

			a = np.random.choice(a.detach().numpy(), p.detach().numpy())
		else:
			a = a.detach().numpy()

		if np.isnan(a):
			print('nan')

		return a


class BetaPolicy(nn.Module):
	def __init__(self, state_dim, action_dim=1, act_min=0, act_max = 1,
				 a_non_linearity=F.relu, a_hidden_layers=1, a_hidden_dim=20,
				 b_non_linearity=F.relu, b_hidden_layers=1, b_hidden_dim=20):

		super(BetaPolicy, self).__init__()

		if not act_min < act_max:
			raise ValueError("The action range is not properly defined: act_min < act_max")

		# Linear transformation from [act_min, act_max] to [0, 1] where the Beta Distribution is defined
		self._action_m = 1 / (act_max - act_min)
		self._action_b = - act_min * self._action_m

		self._state_dim = state_dim
		self._action_dim = action_dim

		#self._beta_func = Beta()

		self._a_non_linearity = a_non_linearity
		self._b_non_linearity = b_non_linearity

		# Variable list of Layers for alpha and beta parameters
		self.a_layers = nn.ModuleList()
		self.b_layers  = nn.ModuleList()

		# Input Layers for alpha and beta parameters
		self.a_layers.append(nn.Linear(state_dim, a_hidden_dim))
		self.b_layers.append(nn.Linear(state_dim, b_hidden_dim))

		# Hidden Layers for alpha and beta parameters
		for i in range(a_hidden_layers):
			self.a_layers.append(nn.Linear(a_hidden_dim, a_hidden_dim))

		for i in range(b_hidden_layers):
			self.b_layers.append(nn.Linear(b_hidden_dim, b_hidden_dim))

		# Output Layers for alpha and beta parameters
		self.a_layers.append(nn.Linear(a_hidden_dim, action_dim))
		self.b_layers.append(nn.Linear(b_hidden_dim, action_dim))

	def forward(self, s, a):

		if not isinstance(s, torch.Tensor):
			s = tt(s)

		if not isinstance(a, torch.Tensor):
			a = tt(a)

		alpha = s
		for i in range(len(self.a_layers) - 1):
			alpha = self._a_non_linearity(self.a_layers[i](alpha))

		# Use softplus to force alpha to be >0
		alpha = F.softplus(self.a_layers[-1](alpha))
		self.alpha = alpha

		beta = s
		for i in range(len(self.b_layers) - 1):
			beta = self._b_non_linearity(self.b_layers[i](beta))

		# Use softplus to force beta to be >0
		beta = F.softplus(self.b_layers[-1](beta))
		self.beta = beta

		# Linearly transforms a continuous action from [act_min, act_max] to [0, 1] where the Beta PDF is defined
		transformed_a = self._action_m * a + self._action_b

		# Beta Distribution

		# Beta(x; alpha, beta) = (1-x)(beta-1) * x ^(alpha-1) / Beta(alpha, beta)
		beta_ab = torch.exp((torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)))
		p = ((1 - transformed_a).pow(beta - 1)) * (transformed_a.pow(alpha - 1)) / beta_ab #self._beta_func(alpha, beta)


		return p


	def get_action(self, s):

		if not hasattr(self, 'alpha') or not hasattr(self, 'beta'):
			self.forward(s, torch.rand(self._action_dim))


		m = torch.distributions.beta.Beta(self.alpha, self.beta)
		a = m.sample()

		if self._action_dim > 1:
			p = self.forward(s, a)

			a = np.random.choice(a.detach().numpy(), p.detach().numpy())
		else:
			a = a.detach().numpy()

		if np.isnan(a):
			print('nan')

		# Transform action back to [act_min, act_max] interval
		a = (a - self._action_b) / self._action_m

		return a