import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple

from utils import *

from value_functions import *


class ReplayBuffer:
	# Replay buffer for experience replay. Stores transitions.
	def __init__(self, max_size):
		self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
		self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
		self._size = 0
		self._max_size = max_size

	def add_transition(self, state, action, next_state, reward, done):


		state      = tn(state)
		action     = tn(action)
		next_state = tn(next_state)
		reward     = tn(reward)
		done       = tn(done)

		self._data.states.append(state)
		self._data.actions.append(action)
		self._data.next_states.append(next_state)
		self._data.rewards.append(reward)
		self._data.terminal_flags.append(done)
		self._size += 1

		if self._size > self._max_size:
			self._data.states.pop(0)
			self._data.actions.pop(0)
			self._data.next_states.pop(0)
			self._data.rewards.pop(0)
			self._data.terminal_flags.pop(0)

	def random_next_batch(self, batch_size):
		batch_indices = np.random.choice(len(self._data.states), batch_size)

		batch_states = np.array([self._data.states[i] for i in batch_indices])
		batch_actions = np.array([self._data.actions[i] for i in batch_indices])
		batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
		batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
		batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])

		return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags.astype(int))


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	soft_update(target, source, 1.0)


class DQN:
	def __init__(self, policy, action_fun, q, q_target, state_dim, action_dim, gamma, doubleQ=True, cuda=False, batch_size=64):

		self._q = q
		self._q_target = q_target

		self._pi = policy
		self._action_fun = action_fun

		self._doubleQ = doubleQ

		if cuda:
			self._q.cuda()
			self._q_target.cuda()

		self._gamma = gamma
		self._loss_function = nn.MSELoss()
		self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
		self._action_dim = action_dim

		self._replay_buffer = ReplayBuffer(1e6)
		self._batch_size = batch_size

	def _get_action(self, s, deterministic=False):
		return self._pi.get_action(s, deterministic=deterministic)

	def get_action(self, s, deterministic=False):
		return self._action_fun.act2env(self._get_action(s, deterministic=deterministic))

	def train(self, env, episodes, time_steps, initial_state=None):

		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

		for e in range(episodes):

			s = env.reset(initial_state=initial_state)
			total_r = 0

			for t in range(time_steps):

				a = self._get_action(s)
				ns, r, d, _ = env.step(self._action_fun.act2env(a))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				total_r += r

				self._replay_buffer.add_transition(s, a, ns, r, d)

				b_states, b_actions, b_nstates, b_rewards, b_terminal = self._replay_buffer.random_next_batch(self._batch_size)

				if self._doubleQ:

					# Action Prediction from Q function
					a_predictions = torch.argmax(self._q(b_nstates), dim=1)
					a_pred_indices = [torch.arange(self._batch_size).long(), a_predictions]

					# Q value from Q_target function using the action indices from Q function
					q_pred = self._q_target(b_nstates)[a_pred_indices]

				else:
					q_pred = torch.max(self._q_target(b_nstates), axis=1)


				target = b_rewards + (1 - b_terminal) * self._gamma * q_pred

				current_prediction = self._q(b_states)[torch.arange(64).long(), b_actions.long()]

				loss = self._loss_function(current_prediction, target.detach())

				self._q_optimizer.zero_grad()
				loss.backward()
				self._q_optimizer.step()

				soft_update(self._q_target, self._q, 0.01)

				if d:
					break
				s = ns

			print("{} Steps in Episode {}/{}. Reward {}".format(t + 1, e + 1, episodes, total_r))

		return stats

	def save(self, dir, file_name):

		if dir != '' and dir[-1] != '/':
			dir = dir + '/'

		torch.save(self._q.state_dict(), '{}{}_q_{}.pt'.format(dir, file_name, timestamp()))
		torch.save(self._q_target.state_dict(), '{}{}_q_target_{}.pt'.format(dir, file_name, timestamp()))