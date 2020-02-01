import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

from value_functions import *
from policies import *

class REINFORCE:
	def __init__(self, policy, action_fun, state_dim, action_dim, gamma, baseline=False, baseline_fun = None):

		self.baseline_fun = baseline_fun
		self._pi = policy
		self._action_fun = action_fun

		self._state_dim  = state_dim
		self._action_dim = action_dim

		self._baseline = baseline

		self._gamma = gamma

		self._create_model()

	def _create_model(self):

		if torch.cuda.is_available():
			self.baseline_fun.cuda()
			self._pi.cuda()

		self._action_indices = np.arange(0, self._action_dim)

		# Optimizers and loss function
		if self.baseline_fun is not None and self._baseline:
			self._V_optimizer = optim.Adam(self.baseline_fun.parameters(), lr=0.0001)
			self._loss_function = nn.MSELoss()

		self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)

	def _get_action(self, s, deterministic=False):

		return self._pi.get_action(s, deterministic=deterministic)

	def get_action(self, s, deterministic=False):
		return self._action_fun.act2env(self._get_action(s, deterministic=deterministic))

	def train(self, env, episodes, time_steps, initial_state=None, initial_noise=0.5):
		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes), episode_loss=np.zeros(episodes))

		for e in range(episodes):
			# Generate an episode.
			# An episode is an array of (state, action, reward) tuples
			episode = []
			s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)

			total_r = 0
			for t in range(time_steps):
				a = self._get_action(s)
				ns, r, d, _ = env.step(self._action_fun.act2env(a))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				episode.append((s, a, r))

				total_r += r

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

				if self._baseline:
					V = self.baseline_fun(s)
					delta = G - V

					v_loss = self._loss_function(G, self.baseline_fun(s))

					self._V_optimizer.zero_grad()
					v_loss.backward()
					self._V_optimizer.step()

					score_fun =  - ((self._gamma ** t) * delta) * torch.log(self._pi(s, a))
				else:
					score_fun = - ((self._gamma ** t) * G) * torch.log(self._pi(s, a))

				stats.episode_loss[e] += score_fun[0].item()

				self._pi_optimizer.zero_grad()
				score_fun.backward()
				self._pi_optimizer.step()

			pr_stats = {'steps': int(stats.episode_lengths[e] + 1), 'episode': e + 1, 'episodes': episodes,
						'reward': stats.episode_rewards[e], 'loss': stats.episode_loss[e]}
			print_stats(pr_stats)

		return stats

	def reset_parameters(self):
		self._pi.reset_parameters()

		if self._baseline:
			self.baseline_fun.reset_parameters()