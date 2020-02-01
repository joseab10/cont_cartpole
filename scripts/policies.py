import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from neural_networks import *
from probability_functions import *

from utils import *



class GaussPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, mean_non_linearity=F.relu, mean_hidden_layers=1, mean_hidden_dim=20,
											   var_non_linearity=F.relu,  var_hidden_layers=1,  var_hidden_dim=20):

		super(GaussPolicy, self).__init__()

		self._state_dim = state_dim
		self._action_dim = action_dim

		self._mean_non_linearity = mean_non_linearity
		self._mean_hidden_layers = mean_hidden_layers
		self._mean_hidden_dim    = mean_hidden_dim

		self._var_non_linearity = var_non_linearity
		self._var_hidden_layers = var_hidden_layers
		self._var_hidden_dim    = var_hidden_dim

		self._create_models()


	def _create_models(self):

		self._mu_fa = MLP(self._state_dim, self._action_dim, hidden_dim=self._mean_hidden_layers,
						  hidden_non_linearity=self._mean_non_linearity, hidden_layers=self._mean_hidden_layers)
		self._sigma_fa = MLP(self._state_dim, self._action_dim, hidden_dim=self._var_hidden_dim,
							 hidden_non_linearity=self._var_non_linearity, hidden_layers=self._var_hidden_layers)

		self._output_layer = NormalDistribution()

	def forward(self, s, a):

		mu = self._mu_fa(s)
		sigma = self._sigma_fa(s)
		p = self._output_layer(a, mu, sigma)

		# Cache last result
		self._p = p

		return p

	def get_action(self, s, deterministic=False):

		# Run to get a fresh value for mu and sigma
		if not hasattr(self, '_p'):
			self.forward(s, torch.rand(self._action_dim))


		if deterministic:

			# Get the point of maximum probability, i.e.: the mode
			a = self._output_layer.mode()
			a = tn(a)

		else:

			a = self._output_layer.sample()

			if self._action_dim > 1:
				a = np.random.choice(tn(a), tn(self._p))
			else:
				a = tn(a)

		return a

	def reset_parameters(self):

		self._mu_fa.reset_parameters()
		self._sigma_fa.reset_parameters()
		self._output_layer.reset_parameters()

	# Methods used for dumping and loading the state using pickle
	def __getstate__(self):

		mu_state = self._mu_fa.__getstate__()
		sigma_state = self._sigma_fa.__getstate__()

		attributes = {
			'_state_dim'         : self._state_dim,
			'_action_dim'        : self._action_dim,

			'_mean_non_linearity': activationFunctionStr(self._mean_non_linearity),
			'_mean_hidden_layers': self._mean_hidden_layers,
			'_mean_hidden_dim'   : self._mean_hidden_dim,

			'_var_non_linearity' : activationFunctionStr(self._var_non_linearity),
			'_var_hidden_layers' : self._var_hidden_layers,
			'_var_hidden_dim'    : self._var_hidden_dim,
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes, 'mu_state': mu_state, 'sigma_state': sigma_state}

		return state_dict

	def __setstate__(self, state):

		super(GaussPolicy, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)

		self._create_models()

		self._mu_fa.__setstate__(state['mu_state'])
		self._sigma_fa.__setstate__(state['sigma_state'])


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

		self._a_non_linearity = a_non_linearity
		self._a_hidden_layers = a_hidden_layers
		self._a_hidden_dim    = a_hidden_dim

		self._b_non_linearity = b_non_linearity
		self._b_hidden_layers = b_hidden_layers
		self._b_hidden_dim    = b_hidden_dim

		self._create_models()

	def _create_models(self):

		# Use softplus to force alpha and beta to be >0
		self._alpha_fa = MLP(self._state_dim, self._action_dim, output_non_linearity=F.softplus,
							 hidden_dim=self._a_hidden_dim, hidden_non_linearity=self._a_non_linearity,
							 hidden_layers=self._a_hidden_layers)
		self._beta_fa = MLP(self._state_dim, self._action_dim, output_non_linearity=F.softplus,
							hidden_dim=self._b_hidden_dim,
							hidden_non_linearity=self._b_non_linearity, hidden_layers=self._b_hidden_layers)

		self._output_layer = BetaDistribution()

	def forward(self, s, a):

		s = tt(s)
		a = tt(a)

		# Linearly transforms a continuous action from [act_min, act_max] to [0, 1] where the Beta PDF is defined
		transformed_a = self._action_m * a + self._action_b

		alpha = self._alpha_fa(s)
		beta  = self._beta_fa(s)
		self._alpha = alpha
		self._beta  = beta

		p = self._output_layer(transformed_a, alpha, beta)
		self._p = p

		return p


	def get_action(self, s, deterministic=False):

		# Run to get a fresh copy of alpha and beta
		self.forward(s, torch.rand(self._action_dim))

		alpha = tn(self._alpha)
		beta = tn(self._beta)


		if deterministic:

			# Get the point of maximum probability, i.e.: the mode
			a = self._output_layer.mode()
			a = tn(a)

		else:
			a = self._output_layer.sample()

			if self._action_dim > 1:
				p = self.forward(s, a)

				a = np.random.choice(tn(a), tn(p))
			else:
				a = tn(a)

		# Transform action back to [act_min, act_max] interval
		a = (a - self._action_b) / self._action_m

		return a

	def reset_parameters(self):

		self._alpha_fa.reset_parameters()
		self._beta_fa.reset_parameters()
		self._output_layer.reset_parameters()


	# Methods used for dumping and loading the state using pickle
	def __getstate__(self):

		alpha_state = self._alpha_fa.__getstate__()
		beta_state = self._beta_fa.__getstate__()

		attributes = {
			'_action_m'       : self._action_m,
			'_action_b'       : self._action_b,

			'_state_dim'      : self._state_dim,
			'_action_dim'     : self._action_dim,

			'_a_non_linearity': activationFunctionStr(self._a_non_linearity),
			'_a_hidden_layers': self._a_hidden_layers,
			'_a_hidden_dim'   : self._a_hidden_dim,

			'_b_non_linearity': activationFunctionStr(self._b_non_linearity),
			'_b_hidden_layers': self._b_hidden_layers,
			'_b_hidden_dim'   : self._b_hidden_dim
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes, 'alpha_state': alpha_state,
					  'beta_state': beta_state}

		return state_dict

	def __setstate__(self, state):

		super(BetaPolicy, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)

		self._create_models()

		self._alpha_fa.__setstate__(state['alpha_state'])
		self._beta_fa.__setstate__(state['beta_state'])


class EpsilonGreedyPolicy:

	def __init__(self, epsilon, value_function):
		self.epsilon = epsilon
		self._value_function = value_function

	def __call__(self, s):
		# Computes the probabilities of taking each action,
		# giving more weight to the action with the best value-function
		value = self._value_function(s)
		value = tn(value)

		nA = len(value)

		p = np.ones(nA, dtype=float) * self.epsilon / (nA - 1)
		best_action = np.argmax(value)
		p[best_action] = 1.0 - self.epsilon

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


	# Methods used for dumping and loading the state using pickle
	def __getstate__(self):

		attributes = {
			'self.epsilon'       : self.epsilon
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes}

		return state_dict

	def __setstate__(self, state):

		super(EpsilonGreedyPolicy, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)