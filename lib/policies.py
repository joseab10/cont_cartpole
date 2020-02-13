import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np

from neural_networks import *
from probability_functions import *

from utils import *


class GaussPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, input_non_linearity=F.relu, input_hidden_layers=1, input_hidden_dim=20):

		super(GaussPolicy, self).__init__()

		self._state_dim = state_dim
		self._action_dim = action_dim

		self._in_non_linearity = input_non_linearity
		self._in_hidden_layers = input_hidden_layers
		self._in_hidden_dim = input_hidden_dim

		self._input_model = MLP(self._state_dim, self._in_hidden_dim, output_non_linearity=None,
								hidden_dim=self._in_hidden_dim, hidden_non_linearity=self._in_non_linearity,
								hidden_layers=self._in_hidden_layers)

		self._mu_model = Linear(self._in_hidden_dim, self._action_dim)

		self._sigma_model = Linear(self._in_hidden_dim, self._action_dim)

		self._output_model = NormalDistribution()

	def forward(self, s, a):

		x = self._input_model(s)

		mu = self._mu_model(x)
		sigma = self._mu_model(x)

		p = self._output_model(a, mu, sigma)

		# Cache last result
		self._p = p

		return p

	def get_action(self, s, deterministic=False):

		# Run to get a fresh value for mu and sigma
		if not hasattr(self, '_p'):
			self.forward(s, torch.rand(self._action_dim))


		if deterministic:

			# Get the point of maximum probability, i.e.: the mode
			a = self._output_model.mode()
			a = tn(a)

		else:

			a = self._output_model.sample()

			if self._action_dim > 1:
				a = np.random.choice(tn(a), tn(self._p))
			else:
				a = tn(a)

		return a

	def reset_parameters(self):

		self._mu_model.reset_parameters()
		self._sigma_model.reset_parameters()
		self._output_model.reset_parameters()

		if hasattr(self, '_p'):
			self.__delattr__('_p')


class BetaPolicy(nn.Module):
	def __init__(self, state_dim, action_dim=1, act_min=0, act_max = 1,
				 input_non_linearity=F.relu, input_hidden_layers=1, input_hidden_dim=20, ns=True):

		super(BetaPolicy, self).__init__()

		if not act_min < act_max:
			raise ValueError("The action range is not properly defined: act_min < act_max")

		self._act_min = act_min
		self._act_max = act_max
		# Linear transformation from [act_min, act_max] to [0, 1] where the Beta Distribution is defined
		self._action_m = 1 / (act_max - act_min)
		self._action_b = - act_min * self._action_m

		self._state_dim = state_dim
		self._action_dim = action_dim

		self._in_non_linearity = input_non_linearity
		self._in_hidden_layers = input_hidden_layers
		self._in_hidden_dim = input_hidden_dim

		self._numerically_stable = ns

		self._input_model = MLP(self._state_dim, self._in_hidden_dim, output_non_linearity=None,
								hidden_dim=self._in_hidden_dim, hidden_non_linearity=self._in_non_linearity,
								hidden_layers=self._in_hidden_layers)

		self._alpha_model = Linear(self._in_hidden_dim, self._action_dim)

		self._beta_model = Linear(self._in_hidden_dim, self._action_dim)

		self._output_model = BetaDistribution()

	def forward(self, s, a):

		s = tt(s)
		a = tt(a)

		x = self._input_model(s)

		# Use softplus to force alpha and beta to be >0
		alpha = self._alpha_model(x)
		alpha = F.softplus(alpha)

		beta = self._beta_model(x)
		beta = F.softplus(beta)

		if self._numerically_stable:
			# Avoid alpha, beta < 1 so the gradient does not go to infinity!
			# We loose deterministic and bimodal PDFs, but it's a price we are willing to pay
			alpha = alpha + 1
			beta  = beta + 1

		self._alpha = alpha
		self._beta  = beta

		# Linearly transforms a continuous action from [act_min, act_max] to [0, 1] where the Beta PDF is defined
		transformed_a = self._action_m * a + self._action_b
		transformed_a = torch.clamp(transformed_a, 0, 1)

		p = self._output_model(transformed_a, alpha, beta)
		self._p = p

		return p


	def get_action(self, s, deterministic=False):

		# Run to get a fresh copy of alpha and beta
		self.forward(s, torch.rand(self._action_dim))

		if deterministic:

			# Get the point of maximum probability, i.e.: the mode
			a = self._output_model.mode()
			a = tn(a)

		else:
			a = self._output_model.sample()

			if self._action_dim > 1:
				p = self.forward(s, a)

				a = np.random.choice(tn(a), tn(p))
			else:
				a = tn(a)

		# Transform action back to [act_min, act_max] interval
		a = (a - self._action_b) / self._action_m
		a = a.clip(self._act_min, self._act_max)

		return a

	def reset_parameters(self):

		self._alpha_model.reset_parameters()
		self._beta_model.reset_parameters()
		self._output_model.reset_parameters()


class MlpPolicy(nn.Module):
	def __init__(self, state_dim, action_dim=1, act_min=0, act_max=1,# act_samples=100,
				 non_linearity=F.relu, hidden_layers=1, hidden_dim=20, output_non_linearity=None,
				 noise=0.2, noise_clip=0.5):

		super(MlpPolicy, self).__init__()

		self._state_dim = state_dim
		self._action_dim = action_dim

		if not act_min < act_max:
			raise ValueError("The action range is not properly defined: act_min < act_max")

		self._act_min = act_min
		self._act_max = act_max

		self._non_linearity     = non_linearity
		self._hidden_layers     = hidden_layers
		self._hidden_dim        = hidden_dim
		self._out_non_linearity = output_non_linearity

		self._noise = noise
		self._noise_clip = noise_clip

		self._fa = MLP(self._state_dim, self._action_dim, self._out_non_linearity, self._hidden_dim,
					   self._non_linearity, self._hidden_layers)

	def forward(self, s, a=None):

		del a

		return self._fa(tt(s))

	def get_action(self, s, deterministic=False):

		a = self.forward(s)

		if not deterministic:
			m = td.normal.Normal(0, self._noise)
			act_noise = m.sample(a.shape)
			act_noise = act_noise.clamp(-self._noise_clip, self._noise_clip)
			act_noise = act_noise.detach()
			a += act_noise

		a = a.clamp(self._act_min, self._act_max)

		return a

	def reset_parameters(self):
		self._fa.reset_parameters()


class EpsilonGreedyPolicy:

	def __init__(self, schedule, value_function):
		self.epsilon = schedule
		self._value_function = value_function


	def step(self):
		self.epsilon.step()

	def __call__(self, s):
		# Computes the probabilities of taking each action,
		# giving more weight to the action with the best value-function
		value = self._value_function(s)
		value = tn(value)

		nA = len(value)

		epsilon = self.epsilon()

		p = np.ones(nA, dtype=float) * epsilon / (nA - 1)
		best_action = np.argmax(value)
		p[best_action] = 1.0 - epsilon

		return p


	def get_action(self, s, deterministic=False):
		# Randomly chooses a single action according to their probabilites

		p = self(s)

		if deterministic:
			a = np.argmax(p)
		else:
			nA = len(p)
			action_indices = np.arange(0, nA)

			a = np.random.choice(action_indices, p=p)

		return a

	def reset_parameters(self):
		self.epsilon.reset_parameters()
