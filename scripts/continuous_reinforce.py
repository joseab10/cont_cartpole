import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

from value_functions import *
from policies import *

class REINFORCE:
	def __init__(self, policy, action_fun, state_dim, action_dim, gamma, baseline=False, baseline_fun = None, lr=1e-4, bl_lr=1e-4):


		self._pi = policy
		self._action_fun = action_fun

		self._state_dim  = state_dim
		self._action_dim = action_dim

		self._baseline = baseline
		self.baseline_fun = baseline_fun
		self._bl_learning_rate = bl_lr

		self._gamma = gamma
		self._learning_rate = lr

		if torch.cuda.is_available():
			self.baseline_fun.cuda()
			self._pi.cuda()

		self._action_indices = np.arange(0, self._action_dim)

		# Optimizers and loss function
		if self.baseline_fun is not None and self._baseline:
			self._bl_optimizer = optim.Adam(self.baseline_fun.parameters(), lr=self._bl_learning_rate)
			self._bl_loss_function = nn.MSELoss()

		self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=self._learning_rate)


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
				ns, r, d, _ = env.step(tn(self._action_fun.act2env(a)))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				episode.append((s, a, r))

				total_r += r

				if d:
					break
				s = ns


			gamma_t = 1
			for t in range(len(episode)):
				# Find the first occurance of the state in the episode
				s, a, r = episode[t]

				G = 0
				gamma_kt = 1
				for k in range(t , len(episode)):
					gamma_kt = gamma_kt * self._gamma
					_, _, r_k = episode[k]
					G = G + (gamma_kt) * r_k

				G = float(G)

				p = self._pi(s, a)

				# For Numerical Stability, in order to not get probabilities higher than one (e.g. delta distribution)
				# and to not return a probability equal to 0 because of the log in the score_function
				eps = 1e-8
				p = p.clamp(eps, 1)

				log_p = torch.log(p)

				gamma_t = gamma_t * self._gamma

				if self._baseline:
					bl = self.baseline_fun(s)
					delta = G - bl

					bl_loss = self._bl_loss_function(self.baseline_fun(s), tt([G]))

					self._bl_optimizer.zero_grad()
					bl_loss.backward()
					self._bl_optimizer.step()

					score_fun = torch.mean(- (gamma_t * delta) * log_p)
				else:
					score_fun = torch.mean(- (gamma_t * G) * log_p)

				stats.episode_loss[e] += score_fun.item()

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