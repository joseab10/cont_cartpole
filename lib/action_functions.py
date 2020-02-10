import numpy as np

import torch

from utils import tn


class ActCont2Cont:
	def __init__(self, act2env_f, env2act_f):
		self._act2env_f = act2env_f
		self._env2act_f = env2act_f

	def act2env(self, a):
		return self._act2env_f(a)

	def env2act(self, a):
		return self._env2act_f(a)


class ActDisc2Cont:
	def __init__(self, action_mapping: dict):

		self._act2env_mapping = action_mapping

		self._env2act_mapping = {}

		for disc, cont in action_mapping.items():
			if cont not in self._env2act_mapping:
				self._env2act_mapping[cont] = disc

	def act2env(self, a):
		return np.array([self._act2env_mapping[a]])

	def env2act(self, a):
		return self._env2act_mapping[a]


def identity(a):
	return tn(a)


def clip_action(a):
	return np.clip(a, -1+1e-8, 1-1e-8)


def clamp_action(min_act, max_act):
	def f(a):
		return torch.clamp(a, min_act, max_act)

	return f


# Convert from discrete to continuous actions through a dictionary mapping
act_disc2cont = ActDisc2Cont({0: -1.00, 1:  1.00})
act_disc3cont = ActDisc2Cont({0: -1.00, 1:  0.00, 2:  1.00})
act_disc4cont = ActDisc2Cont({0: -1.00, 1: -0.50, 2:  0.50, 3:  1.00})
act_disc5cont = ActDisc2Cont({0: -1.00, 1: -0.50, 2:  0.00, 3:  0.50, 4: 1.00})
act_disc8cont = ActDisc2Cont({0: -1.00, 1: -0.75, 2: -0.50, 3: -0.25, 4: 0.25, 5: 0.50, 6: 0.75, 7: 1.00})
act_disc9cont = ActDisc2Cont({0: -1.00, 1: -0.75, 2: -0.50, 3: -0.25, 4: 0.00, 5: 0.25, 6: 0.50, 7: 0.75, 8: 1.00})
# Clips actions between [-1, 1]
act_clipcont  = ActCont2Cont(clip_action, clip_action)
act_identity  = ActCont2Cont(identity, identity)
