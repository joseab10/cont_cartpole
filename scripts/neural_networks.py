import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tt


def getActivationFunction(af):

	if af is None:
		return af

	elif isinstance(af, str):
		if af == 'None':
			return None

		elif hasattr(torch, af):
			return getattr(torch, af)

		elif hasattr(F, af):
			return getattr(F, af)

		else:
			raise NotImplemented('Pytorch Functional does not have ' + af + ' implemented yet')

	elif callable(af):
		return af

	else:
		raise TypeError('Invalid Activation Function!')

def activationFunctionStr(af):
	if af is None:
		return 'None'
	elif isinstance(af, str):
		return af
	else:
		return af.__name__



class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, output_non_linearity=None,
				 hidden_dim=20, hidden_non_linearity=F.relu, hidden_layers=1):

		super(MLP, self).__init__()

		self._hidden_layers = hidden_layers

		# Layer Dimensions
		self._input_dim = input_dim
		self._hidden_dim = hidden_dim
		self._output_dim = output_dim

		# Activation Functions
		self._hidden_non_linearity = hidden_non_linearity
		self._output_non_linearity = output_non_linearity

		self._create_model()


	def _create_model(self):

		# Parse Activation Functions
		self._hidden_non_linearity = getActivationFunction(self._hidden_non_linearity)
		self._output_non_linearity = getActivationFunction(self._output_non_linearity)

		# Variable list of Layers
		self.layers = nn.ModuleList()

		# Input Layer
		self.layers.append(nn.Linear(self._input_dim, self._hidden_dim))

		# Hiden Layers
		for i in range(self._hidden_layers):
			self.layers.append(nn.Linear(self._hidden_dim, self._hidden_dim))

		# Output Layers
		self.layers.append(nn.Linear(self._hidden_dim, self._output_dim))

	def forward(self, x):

		if not isinstance(x, torch.Tensor):
			x = tt(x)

		for i in range(len(self.layers) - 1):
			x = self.layers[i](x)
			if self._hidden_non_linearity is not None:
				x = self._hidden_non_linearity(x)

		x = self.layers[-1](x)

		if self._output_non_linearity is not None:
			x = self._output_non_linearity(x)

		return x


	def reset_parameters(self):

		for layer in self.layers:
			layer.reset_parameters()


	# Methods used for dumping and loading the state using pickle
	def __getstate__(self):

		model_state = self.state_dict()

		attributes = {
			'_input_dim' 			: self._input_dim,
			'_output_dim'			: self._output_dim,
			'_output_non_linearity'	: activationFunctionStr(self._output_non_linearity),
			'_hidden_dim'			: self._hidden_dim,
			'_hidden_non_linearity' : activationFunctionStr(self._hidden_non_linearity),
			'_hidden_layers'		: self._hidden_layers
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes, 'model_state': model_state}

		return state_dict

	def __setstate__(self, state):

		super(MLP, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)

		self._create_model()

		self.load_state_dict(state['model_state'])


class Linear(nn.Module):
	def __init__(self, input_dim, output_dim):

		super(Linear, self).__init__()

		# Layer Dimensions
		self._input_dim = input_dim
		self._output_dim = output_dim

		self._create_model()

	def _create_model(self):

		# Linear Layer
		self._fc1 = nn.Linear(self._input_dim, self._output_dim)


	def forward(self, x):

		if not isinstance(x, torch.Tensor):
			x = tt(x)

		x = self._fc1(x)

		return x

	def reset_parameters(self):
		self._fc1.reset_parameters()


	# Methods used for dumping and loading the state using pickle
	def __getstate__(self):

		model_state = self.state_dict()

		attributes = {
			'_input_dim' 			: self._input_dim,
			'_output_dim'			: self._output_dim,
		}

		state_dict = {'class': type(self).__name__, 'attributes': attributes, 'model_state': model_state}

		return state_dict

	def __setstate__(self, state):

		super(Linear, self).__init__()

		for att, value in state['attributes'].items():
			setattr(self, att, value)

		self._create_model()

		self.load_state_dict(state['model_state'])