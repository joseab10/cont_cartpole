import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

	#def backward(self, gradient):

	#	beta_grad_a, beta_grad_b = self._beta_func.backward(gradient)

	#	return beta_grad_a * self.alpha.backward(), beta_grad_b * self.beta.backward()

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



class REINFORCE:
	def __init__(self, env, state_dim, action_dim, gamma, cuda=False):

		self._V = StateValueFunction(state_dim)
		self._pi = BetaPolicy(state_dim, action_dim, act_min=-1, act_max=1)

		self.env = env

		if cuda:
			self._V.cuda()
			self._pi.cuda()

		self._gamma = gamma
		self._loss_function = nn.MSELoss()
		self._V_optimizer = optim.Adam(self._V.parameters(), lr=0.0001)
		self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)
		self._action_dim = action_dim
		self._action_indices = np.arange(0, action_dim)
		self._loss_function = nn.MSELoss()

	def get_action(self, s):

		return np.clip(self._pi.get_action(s), -1, 1)

	def train(self, episodes, time_steps, baseline=False):
		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

		for i_episode in range(1, episodes + 1):
			# Generate an episode.
			# An episode is an array of (state, action, reward) tuples
			episode = []
			s = self.env.reset()
			for t in range(time_steps):
				a = self.get_action(s)
				ns, r, d, _ = self.env.step(a)

				stats.episode_rewards[i_episode - 1] += r
				stats.episode_lengths[i_episode - 1] = t

				episode.append((s, a, r))

				if d:
					break
				s = ns


			for t in range(len(episode)):
				# Find the first occurance of the state in the episode
				s, a, r = episode[t]

				G = 0
				for k in range(t + 1, len(episode)):
					_, _, r_k = episode[k]
					G = G + (self._gamma ** (k - t - 1)) * r_k

				G = float(G)

				if baseline:
					V = self._V(s)
					delta = G - V

					v_loss = self._loss_function(G, self._V(s))

					self._V_optimizer.zero_grad()
					v_loss.backward()
					self._V_optimizer.step()

					score_fun =  - ((self._gamma ** t) * delta) * torch.log(self._pi(s, a))
				else:
					score_fun = - ((self._gamma ** t) * G) * torch.log(self._pi(s, a))

				self._pi_optimizer.zero_grad()
				score_fun.backward()
				self._pi_optimizer.step()



			print("\r{} Steps in Episode {}/{}. Reward {}".format(len(episode), i_episode, episodes,
																  sum([e[2] for i, e in enumerate(episode)])))
		return stats