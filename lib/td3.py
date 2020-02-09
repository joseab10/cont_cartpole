import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from policies import MlpPolicy
from neural_networks import MLP
from deep_q_learning import soft_update, hard_update
from buffer import ReplayBuffer
from utils import tt, tn, EpisodeStats, print_stats


class MlpActor(MlpPolicy):
	def __init__(self, state_dim, action_dim=1, act_min=0, act_max=1,# act_samples=100,
				 non_linearity=F.relu, hidden_layers=1, hidden_dim=20, output_non_linearity=F.sigmoid,
				 noise=0.2, noise_clip=0.5):

		super(MlpActor, self).__init__(state_dim, action_dim, act_min, act_max,# act_samples,
									non_linearity, hidden_layers, hidden_dim, output_non_linearity,
									   noise=noise, noise_clip=noise_clip)


class MlpCritic(nn.Module):
	def __init__(self, state_dim, action_dim,
				 non_linearity=F.relu, hidden_layers=1, hidden_dim=20, output_non_linearity=F.sigmoid):
		super(MlpCritic, self).__init__()

		input_dim = state_dim + action_dim

		# Q1 FA
		self._Q1 = MLP(input_dim=input_dim, output_dim=1, output_non_linearity=output_non_linearity,
				 hidden_dim=hidden_dim, hidden_non_linearity=non_linearity, hidden_layers=hidden_layers)

		# Q2 FA
		self._Q2 = MLP(input_dim=input_dim, output_dim=1, output_non_linearity=output_non_linearity,
				 hidden_dim=hidden_dim, hidden_non_linearity=non_linearity, hidden_layers=hidden_layers)

	def forward(self, s, a):
		s = tt(s)
		a = tt(a)

		if len(s.shape) == 1:
			x = torch.cat((s, a))
		else:
			x = torch.cat((s, a), dim=1)

		q1 = self._Q1(x)
		q2 = self._Q2(x)

		return q1, q2

	def Q1(self, s, a):
		s = tt(s)
		a = tt(a)

		if len(s.shape) == 1:
			x = torch.cat((s, a))
		else:
			x = torch.cat((s, a), dim=1)

		q1 = self._Q1(x)

		return q1

	def reset_parameters(self):
		self._Q1.reset_parameters()
		self._Q2.reset_parameters()


class TD3:
	def __init__(self, actor, critic, reward_fun, gamma=0.99, tau=0.005, # policy_noise=0.2, noise_clip=0.5,
				 policy_freq=2, max_buffer_size=1e6, batch_size=64, lr=3e-4
				 ):

		self._actor = actor
		self._actor_target = copy.deepcopy(self._actor)
		self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=lr)

		self._critic = critic
		self._critic_target = copy.deepcopy(self._critic)
		self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=lr)

		self.reward_fun = reward_fun

		self._gamma = gamma
		self._tau = tau
		self.policy_freq = policy_freq

		self._rbuffer_max_size = max_buffer_size
		self._replay_buffer = ReplayBuffer(self._rbuffer_max_size)
		self._batch_size = batch_size

		self._steps = 0

	def get_action(self, s, deterministic=False):
		return self._actor.get_action(s, deterministic=deterministic)

	def train(self, env, episodes, time_steps, initial_state=None, initial_noise=0.5):

		stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes),
							 episode_loss=np.zeros(episodes))

		for e in range(episodes):

			s = env.reset(initial_state=initial_state, noise_amplitude=initial_noise)

			for t in range(time_steps):

				a = self.get_action(s)
				ns, r, d, _ = env.step(tn(a))

				stats.episode_rewards[e] += r
				stats.episode_lengths[e] = t

				self._steps += 1
				self._replay_buffer.add_transition(s, a, ns, r, d)

				# Sample replay buffer
				b_states, b_actions, b_nstates, b_rewards, b_terminal = self._replay_buffer.random_next_batch(self._batch_size)

				# Get action according to target actor policy
				b_nactions = self._actor_target(b_nstates)

				# Compute the target Q value from target critic
				target_Q1, target_Q2 = self._critic_target(b_nstates, b_nactions)
				target_Q = torch.min(target_Q1, target_Q2).reshape((-1))
				target_Q = b_rewards + (1 - b_terminal) * self._gamma * target_Q
				target_Q = target_Q.reshape((-1,1)).detach()

				# Get current Q estimates from critic
				current_Q1, current_Q2 = self._critic(b_states, b_actions)

				# Compute critic loss
				critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

				stats.episode_loss[e] += critic_loss.item()

				# Optimize the critic
				self._critic_optimizer.zero_grad()
				critic_loss.backward()
				self._critic_optimizer.step()

				# Delayed policy updates
				if self._steps % self.policy_freq == 0:

					# Compute actor losses
					actor_loss = -self._critic.Q1(b_states, self._actor(b_states)).mean()

					# Optimize the actor
					self._actor_optimizer.zero_grad()
					actor_loss.backward()
					self._actor_optimizer.step()

					# Soft-Update the target models
					soft_update(self._critic_target, self._critic, self._tau)
					soft_update(self._actor, self._actor_target, self._tau)

				if d:
					break
				s = ns

			pr_stats = {'steps': int(stats.episode_lengths[e] + 1), 'episode': e + 1, 'episodes': episodes,
						'reward': stats.episode_rewards[e], 'loss': stats.episode_loss[e]}
			print_stats(pr_stats)

		return stats

	def reset_parameters(self):
		self._actor.reset_parameters()
		self._actor_target.reset_parameters()
		self._critic.reset_parameters()
		self._critic_target.reset_parameters()
		self._steps = 0
		self._replay_buffer = ReplayBuffer(self._rbuffer_max_size)
