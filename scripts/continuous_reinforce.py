import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

from value_functions import *
from policies import *

class REINFORCE:
	def __init__(self, policy, action_fun, state_dim, action_dim, gamma, baseline=False, V = None, cuda=False):

		self._V = V
		self._pi = policy
		self._action_fun = action_fun

		self._baseline = baseline

		if cuda:
			self._V.cuda()
			self._pi.cuda()

		self._gamma = gamma
		self._loss_function = nn.MSELoss()

		if V is not None and baseline:
			self._V_optimizer = optim.Adam(self._V.parameters(), lr=0.0001)

		self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)
		self._action_dim = action_dim
		self._action_indices = np.arange(0, action_dim)
		self._loss_function = nn.MSELoss()

	def _get_action(self, s, deterministic=False):

		return self._pi.get_action(s, deterministic=deterministic)

	def get_action(self, s, deterministic=False):
		return self._action_fun.act2env(self._get_action(s, deterministic=deterministic))

	def train(self, env, episodes, time_steps, initial_state=None):
		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

		for i_episode in range(1, episodes + 1):
			# Generate an episode.
			# An episode is an array of (state, action, reward) tuples
			episode = []
			s = env.reset(initial_state=initial_state)

			total_r = 0
			for t in range(time_steps):
				a = self._get_action(s)
				ns, r, d, _ = env.step(self._action_fun.act2env(a))

				stats.episode_rewards[i_episode - 1] += r
				stats.episode_lengths[i_episode - 1] = t

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



			print("\r{} Steps in Episode {}/{}. Reward {}".format(len(episode), i_episode, episodes, total_r))
		return stats

	def save(self, dir, file_name):

		if dir != '' and dir[-1] != '/':
			dir = dir + '/'

		if self._baseline:
			torch.save(self._V.state_dict(), '{}{}_V_{}.pt'.format(dir, file_name, timestamp()))
		torch.save(self._pi.state_dict(), '{}{}_pi_{}.pt'.format(dir, file_name, timestamp()))