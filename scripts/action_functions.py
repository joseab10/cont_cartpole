import numpy as np

class ActCont2Cont:
	def __init__(self, act2env_f, env2act_f):
		self._act2env_f = act2env_f
		self._env2act_f = env2act_f

	def act2env(self, a):
		return self._act2env_f(a)

	def env2act(self, a):
		return self._env2act_f(a)



class ActDisc2Cont:
	def __init__(self, action_mapping:dict):

		self._act2env_mapping = action_mapping

		self._env2act_mapping = {}

		for disc, cont in action_mapping.items():
			if cont not in self._env2act_mapping:
				self._env2act_mapping[cont] = disc

	def act2env(self, a):
		return np.array([self._act2env_mapping[a]])

	def env2act(self, a):
		return self._env2act_mapping[a]