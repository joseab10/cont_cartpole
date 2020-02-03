import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple

from utils import tn, EpisodeStats, print_stats, tt


class ReplayBuffer:
	# Replay buffer for experience replay. Stores transitions.
	def __init__(self, max_size):

		self._max_size = max_size
		self._create()

	def _create(self):
		self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
		self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
		self._size = 0

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

	def __getstate__(self):

		# Get rid of all data when dumping agents
		attributes = {
			'_max_size': self._max_size
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes}

		return state_dict

	def __setstate__(self, state):

		super(ReplayBuffer, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	soft_update(target, source, 1.0)


class DQN:
	def __init__(self, policy, action_fun, q, q_target, state_dim, action_dim, gamma, doubleQ=True, max_buffer_size=1e6, batch_size=64, lr=1e-4):

		self._q = q
		self._q_target = q_target

		self._pi = policy
		self._action_fun = action_fun

		self._doubleQ = doubleQ

		if torch.cuda.is_available():
			self._q.cuda()
			self._q_target.cuda()

		self._gamma = gamma


		self._state_dim  = state_dim
		self._action_dim = action_dim

		self._rbuffer_max_size = max_buffer_size
		self._replay_buffer = ReplayBuffer(self._rbuffer_max_size)
		self._batch_size = batch_size
		self._learning_rate = lr

		self._loss_function = nn.MSELoss()
		self._q_optimizer = optim.Adam(self._q.parameters(), lr=self._learning_rate)

	def _get_action(self, s, deterministic=False):
		return self._pi.get_action(s, deterministic=deterministic)

	def get_action(self, s, deterministic=False):
		return self._action_fun.act2env(self._get_action(s, deterministic=deterministic))

	def train(self, env, episodes, time_steps, initial_state=None, initial_noise=0.5):

		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes), episode_loss=np.zeros(episodes))

		for e in range(episodes):

			s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)
			total_r = 0
			total_episode_loss = 0

			for t in range(time_steps):

				a = self._get_action(s)
				ns, r, d, _ = env.step(self._action_fun.act2env(a))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				total_r += r

				self._replay_buffer.add_transition(s, a, ns, r, d)

				b_states, b_actions, b_nstates, b_rewards, b_terminal = self._replay_buffer.random_next_batch(self._batch_size)

				if self._doubleQ:

					# Q-Values from next states [Q] used only to determine the optima next actions
					q_nstates = self._q(b_nstates)
					# Optimal Action Prediction  [Q]
					nactions = torch.argmax(q_nstates, dim=1)
					nactions_indices = [torch.arange(self._batch_size).long(), nactions]

					# Q-Values from [Q_target] function using the action indices from [Q] function
					q_target_nstates = self._q_target(b_nstates)[nactions_indices]

				else:
					q_target_nstates = self._q_target(b_nstates)
					q_target_nstates = torch.max(q_target_nstates, axis=1)


				target_prediction = b_rewards + (1 - b_terminal) * self._gamma * q_target_nstates

				current_prediction = self._q(b_states)[torch.arange(self._batch_size).long(), b_actions.long()]

				loss = self._loss_function(current_prediction, target_prediction.detach())

				stats.episode_loss[e] += loss.item()

				self._q_optimizer.zero_grad()
				loss.backward()
				self._q_optimizer.step()

				soft_update(self._q_target, self._q, 0.01)

				if d:
					break
				s = ns

			pr_stats = {'steps': int(stats.episode_lengths[e] + 1), 'episode': e + 1, 'episodes': episodes,
						'reward': stats.episode_rewards[e], 'loss': stats.episode_loss[e]}
			print_stats(pr_stats)


		return stats


	def reset_parameters(self):
		self._q.reset_parameters()
		self._q_target.reset_parameters()
		self._replay_buffer = ReplayBuffer(self._rbuffer_max_size)