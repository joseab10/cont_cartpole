import numpy as np

from collections import namedtuple
from utils import tt, tn


class ReplayBuffer:
	# Replay buffer for experience replay. Stores transitions.
	def __init__(self, max_size):

		self._max_size = max_size

		self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
		self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
		self._size = 0

		self._min_r = np.inf
		self._max_r = -np.inf
		self._sum_r = 0

	def add_transition(self, state, action, next_state, reward, done):

		state      = tn(state)
		action     = tn(action)
		next_state = tn(next_state)
		reward     = tn(reward)
		done       = tn(done)

		self._sum_r += reward

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
			self._size -= 1

	def random_next_batch(self, batch_size):

		batch_indices = np.random.choice(len(self._data.states), batch_size)

		batch_states = np.array([self._data.states[i] for i in batch_indices])
		batch_actions = np.array([self._data.actions[i] for i in batch_indices])
		batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
		batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
		batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])

		nb = (tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards),
			  tt(batch_terminal_flags.astype(int)))

		return nb

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
