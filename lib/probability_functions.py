import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np

from utils import tt


class NormalDistribution(nn.Module):

	def __init__(self):

		super(NormalDistribution, self).__init__()

		self.reset_parameters()

	def forward(self, x, mu, sigma):

		x     = tt(x)
		mu    = tt(mu)
		sigma = tt(sigma)

		self._mu    = mu
		self._sigma = sigma


		p = 1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp((-1/2) * (torch.div(mu - x, sigma)**2))

		return p


	def sample(self):

		m = td.normal.Normal(self._mu, self._sigma)
		x = m.sample()

		return x

	def mean(self):
		return self._mu

	def mode(self):
		return self.mean()

	def variance(self):
		return self._sigma ** 2

	def reset_parameters(self):
		self._mu = 0
		self._sigma = 1


class BetaDistribution(nn.Module):
	def __init__(self):
		super(BetaDistribution, self).__init__()

		self.reset_parameters()

	def forward(self, x, alpha, beta):
		x     = tt(x)
		alpha = tt(alpha)
		beta  = tt(beta)

		self._alpha = alpha
		self._beta  = beta

		beta_ab = torch.exp((torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)))

		p = (torch.pow(x, alpha - 1) * torch.pow(1 - x, beta - 1)) / beta_ab

		return p

	def sample(self):
		m = td.beta.Beta(self._alpha, self._beta)
		x = m.sample()

		return x

	def mean(self):
		return self._alpha / (self._alpha + self._beta)

	def mode(self):

		alpha = self._alpha.detach().numpy()
		beta  = self._beta.detach().numpy()

		mode = np.zeros(alpha.shape[0])

		indices = np.arange(0, mode.shape[0])

		idx = indices[(alpha > 1) & (beta > 1)]
		mode[idx] = (alpha[idx] - 1) / (alpha[idx] + beta[idx] - 2)

		# Uniform
		idx = indices[(alpha == 1) & (beta == 1)]
		mode[idx] = np.random.uniform(0, 1, len(idx))

		# Bi-Modal
		idx = indices[(alpha < 1) & (beta < 1)]
		mode[idx] = np.random.choice([0, 1], len(idx))

		idx = indices[(alpha <= 1) & (beta > 1)]
		mode[idx] = 0

		idx = indices[(alpha > 1) & (beta <= 1)]
		mode[idx] = 1

		return tt(mode)

	def variance(self):
		return (self._alpha * self._beta) / (((self._alpha + self._beta)**2) * (self._alpha + self._beta + 1))

	def reset_parameters(self):
		self._alpha = 2
		self._beta  = 2