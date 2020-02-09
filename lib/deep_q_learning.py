import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import tn, EpisodeStats, print_stats, tt
from buffer import ReplayBuffer


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	soft_update(target, source, 1.0)


class DQN:
	def __init__(self, policy, action_fun, q, q_target, state_dim, action_dim, gamma, double_q=True, reward_fun=None,
					replay_buffer=False, max_buffer_size=1e6, batch_size=64, tau=0.01,
					lr=1e-4):

		self._q = q
		self._q_target = q_target

		self._pi = policy
		self._action_fun = action_fun

		self.reward_fun = reward_fun

		self._doubleQ = double_q

		if torch.cuda.is_available():
			self._q.cuda()
			self._q_target.cuda()

		self._gamma = gamma
		self._tau = tau

		self._state_dim  = state_dim
		self._action_dim = action_dim

		self._use_rbuffer = replay_buffer
		if self._use_rbuffer:
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

		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes),
							 episode_loss=np.zeros(episodes))

		for e in range(episodes):

			s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)
			total_r = 0

			# Step policy for advancing the scheduler
			epsilon = self._pi.epsilon()
			# print("\t\t\tStep: {:5d} Epsilon: {:6.5f}".format(t, epsilon))
			self._pi.step()

			for t in range(time_steps):

				a = self._get_action(s)
				ns, r, d, _ = env.step(self._action_fun.act2env(a))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				total_r += r

				if self._use_rbuffer:
					self._replay_buffer.add_transition(s, a, ns, r, d)
					b_states, b_actions, b_nstates, b_rewards, b_terminal = self._replay_buffer.random_next_batch(self._batch_size)
					dim = 1
				else:
					b_states = s
					b_actions = a
					b_nstates = ns
					b_rewards = r
					b_terminal = d
					dim = 0

				if self._doubleQ:

					# Q-Values from next states [Q] used only to determine the optima next actions
					q_nstates = self._q(b_nstates)
					# Optimal Action Prediction  [Q]
					nactions = torch.argmax(q_nstates, dim=dim)
					if self._use_rbuffer:
						nactions = [torch.arange(self._batch_size).long(), nactions]

					# Q-Values from [Q_target] function using the action indices from [Q] function
					q_target_nstates = self._q_target(b_nstates)[nactions]

				else:
					q_target_nstates = self._q_target(b_nstates)
					q_target_nstates = torch.max(q_target_nstates, dim=dim)

				target_prediction = b_rewards + (1 - b_terminal) * self._gamma * q_target_nstates

				if self._use_rbuffer:
					q_actions = [torch.arange(self._batch_size).long(), b_actions.long()]
				else:
					q_actions = b_actions

				current_prediction = self._q(b_states)[q_actions]

				loss = self._loss_function(current_prediction, target_prediction.detach())

				stats.episode_loss[e] += loss.item()

				self._q_optimizer.zero_grad()
				loss.backward()
				self._q_optimizer.step()

				soft_update(self._q_target, self._q, self._tau)

				if d:
					break
				s = ns

			pr_stats = {'steps': int(stats.episode_lengths[e] + 1), 'episode': e + 1, 'episodes': episodes,
						'reward': stats.episode_rewards[e], 'loss': stats.episode_loss[e]}
			print_stats(pr_stats, ', Epsilon: {:6.5f}'.format(epsilon))

		return stats

	def reset_parameters(self):
		self._q.reset_parameters()
		self._q_target.reset_parameters()
		self._pi.reset_parameters()
		if self._use_rbuffer:
			self._replay_buffer = ReplayBuffer(self._rbuffer_max_size)
