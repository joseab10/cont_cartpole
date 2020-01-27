import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

from value_functions import *
from policies import *

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