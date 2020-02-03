import numpy as np

from continuous_reinforce import REINFORCE
from deep_q_learning import DQN
from policies import *
from action_functions import *
from value_functions import *

from collections import OrderedDict
import json


#================================================================================
#
# "Public" Methods
#
#================================================================================

def parse_reward_function(key, refs):
	return _para_parse_obj(key, {}, _para_reward_func_dict, refs, addref=False)

def parse_init_state(key):
	refs = {}
	return _para_parse_primitive(key, {}, _para_init_states_dict, refs, addref=False)

def parse_init_noise(key):
	refs = {}
	return _para_parse_primitive(key, {}, _para_init_noises_dict, refs, addref=False)


#================================================================================
#
# Type parsing functions
#
#================================================================================

def _para_add_ref(refs, key, value, addref=True):
	if addref:
		refs[key] = value

def _para_parse_primitive(key, args, dict, refs, addref=True):
	if key in args:
		value = args[key]
	else:
		value = dict['default']

	_para_add_ref(refs, key, value, addref=addref)

	return value

def _para_parse_dict(key, args, dict, refs, addref=True):

	value = dict['default']

	if key in args:
		if args[key] in dict:
			value = dict[args[key]]

	_para_add_ref(refs, key, value, addref=addref)

	return value

def _para_parse_obj(key, args, dict, refs, addref=True):

	args_subdict = _para_get_subdict(args, key)
	args_class_name = _para_get_param(args_subdict, 'class')

	obj_class_dict = _para_parse_primitive(args_class_name, dict, dict, refs, addref=False)
	obj_class = _para_get_param(obj_class_dict, 'class')

	if obj_class is None:
		return None

	constructor_args = _para_parse_args(None, _para_get_subdict(args_subdict, 'args'), obj_class_dict['args'], refs)

	obj = obj_class(**constructor_args)

	_para_add_ref(refs, key, obj, addref=addref)

	return obj

def _para_parse_ref(key, value, dict, refs):

	ref_key    = _para_get_param(dict, 'key')
	ref_object = _para_get_param(refs, ref_key, None)
	ref_spath  = _para_get_param(dict, 'subpath')

	if ref_spath != '':
		ref_spath = ref_spath.split('.')
	else:
		ref_spath = []

	while ref_object is not None and len(ref_spath) > 0:
		if hasattr(ref_object, ref_spath[0]):
			ref_object = getattr(ref_object, ref_spath[0])
			ref_spath = ref_spath[1:]
		else:
			ref_object = None

	return ref_object


#================================================================================
#
# Parameter Dictionaries, objects, functions default values and
# other data structures
#
#================================================================================

# Argument Parsing
#--------------------------------------------------------------------------------
_arg_func_obj = {'func': _para_parse_obj, 'args': 'dict'}
_arg_func_dict  = {'func': _para_parse_dict, 'args': 'dict'}
_arg_func_prim = {'func': _para_parse_primitive}
_arg_func_ref  = {'func': _para_parse_ref, 'args': 'dict'}

_arg_types = {
	'obj': _arg_func_obj,
	'ref': _arg_func_ref,
	'dict' : _arg_func_dict,
	'int': _arg_func_prim, 'float': _arg_func_prim, 'str': _arg_func_prim, 'bool': _arg_func_prim,
}

# Initial State
#--------------------------------------------------------------------------------
init_state_up = np.zeros(4)

_para_init_states_dict = {
	'0': None, 'no': None, 'none': None, 'default': None,
	'1': init_state_up, 'up': init_state_up, 'upright': init_state_up
}


# Initial Noise
#--------------------------------------------------------------------------------
init_noise_360 = np.array([0.5, 0.5, np.pi, 0.5])

_para_init_noises_dict = {
	'0': 0, 'det': 0, 'deterministic': 0, 'no': 0, 'none': 0,
	'1': None, 'default': None,
	'2': init_noise_360, '360_deg': init_noise_360, '360': init_noise_360
}


# Reward Functions
#--------------------------------------------------------------------------------
def informative_reward_generator(time_steps):

	def informative_reward(cart_pole):

		cos_pow = 3
		max_pts = 100

		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -max_pts
		else:
			return (np.cos(cart_pole.state[2])**cos_pow)*(max_pts/(2 * time_steps))

	return informative_reward

_para_none_obj_dict = {
	#'type' : 'obj',
	'class': None,
	'args' : {}
}
_para_rf_informative_dict = {
	#'type' : 'obj',
	'class': informative_reward_generator,
	'args' :{
		'time_steps': {'type': 'ref', 'key': 'time_steps', 'subpath': ''}
	}
}

_para_reward_func_dict = {
	'0': _para_none_obj_dict, 'no': _para_none_obj_dict, 'default': _para_none_obj_dict,
	'1': _para_rf_informative_dict, 'info': _para_rf_informative_dict, 'informative': _para_rf_informative_dict
}


# Action Functions
#--------------------------------------------------------------------------------
def ident(a):
	return tn(a)

def clip_action(a):
	return np.clip(a, -1+1e-8, 1-1e-8)

# Convert from discrete to continuous actions through a dictionary mapping
act_disc2cont = ActDisc2Cont({0: -1.00, 1:  1.00})
act_disc3cont = ActDisc2Cont({0: -1.00, 1:  0.00, 2:  1.00})
act_disc4cont = ActDisc2Cont({0: -1.00, 1: -0.50, 2:  0.50, 3:  1.00})
act_disc5cont = ActDisc2Cont({0: -1.00, 1: -0.50, 2:  0.00, 3:  0.50, 4: 1.00})
act_disc8cont = ActDisc2Cont({0: -1.00, 1: -0.75, 2: -0.50, 3: -0.25, 4: 0.25, 5: 0.50, 6: 0.75, 7: 1.00})
act_disc9cont = ActDisc2Cont({0: -1.00, 1: -0.75, 2: -0.50, 3: -0.25, 4: 0.00, 5: 0.25, 6: 0.50, 7: 0.75, 8: 1.00})
# Clips actions between [-1, 1]
act_clipcont  = ActCont2Cont(clip_action, clip_action)
act_ident     = ActCont2Cont(ident, ident)

_para_action_fun_dict = {
	'default': act_ident,
	'clip' : act_clipcont ,
	'2' : act_disc2cont, 'disc2': act_disc2cont,
	'3' : act_disc3cont, 'disc3': act_disc3cont,
	'4' : act_disc4cont, 'disc4': act_disc4cont,
	'5' : act_disc5cont, 'disc5': act_disc5cont,
	'8' : act_disc8cont, 'disc8': act_disc8cont,
	'9' : act_disc9cont, 'disc9': act_disc9cont
}

# Policies
#--------------------------------------------------------------------------------
_para_pi_gaussian_dict = {
	'class': GaussPolicy,
	'args' : {
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'mean_non_linearity' : {'type': 'str', 'default': 'relu'},
		'mean_hidden_layers' : {'type': 'int', 'default': 1},
		'mean_hidden_dim'    : {'type': 'int', 'default': 20},

		'var_non_linearity'  : {'type': 'str', 'default': 'relu'},
		'var_hidden_layers'  : {'type': 'int', 'default': 1},
		'var_hidden_dim'     : {'type': 'int', 'default': 20},
	}
}

_para_pi_beta_dict = {
	'class': BetaPolicy,
	'args' : {
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'act_min': {'type': 'float', 'default': -1},
		'act_max': {'type': 'float', 'default': 1},

		'a_non_linearity': {'type': 'str', 'default': 'relu'},
		'a_hidden_layers': {'type': 'int', 'default': 1},
		'a_hidden_dim'   : {'type': 'int', 'default': 20},

		'b_non_linearity': {'type': 'str', 'default': 'relu'},
		'b_hidden_layers': {'type': 'int', 'default': 1},
		'b_hidden_dim'   : {'type': 'int', 'default': 20},

		'ns': {'type': 'bool', 'default': True}
	}
}

_para_pi_mlp_dict = {
	'class': MlpPolicy,
	'args' : {
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'act_min'    : {'type': 'float', 'default': -1},
		'act_max'    : {'type': 'float', 'default': 1},
		'act_samples': {'type': 'int', 'default': 100},

		'non_linearity'        : {'type': 'str', 'default': 'relu'},
		'hidden_layers'        : {'type': 'int', 'default': 1},
		'hidden_dim'           : {'type': 'int', 'default': 20},
		'output_non_linearity' : {'type': 'str', 'default': 'sigmoid'}
	}
}

_para_pi_egreedy_dict = {
	'class': EpsilonGreedyPolicy,
	'args' : {
		'epsilon': {'type': 'float', 'default': 0.20},

		'value_function': {'type': 'ref', 'key': 'q', 'subpath': ''}
	}
}
_para_policies_reinforce_dict = {
	'default' : _para_pi_beta_dict,
	'gaussian': _para_pi_gaussian_dict,
	'beta'    : _para_pi_beta_dict,
	'mlp'     : _para_pi_mlp_dict
}

_para_policies_dqn_dict = {
	'default': _para_pi_egreedy_dict,
	'epsilon_greedy': _para_pi_egreedy_dict,
	'egreedy': _para_pi_egreedy_dict
}


# Value Functions
#--------------------------------------------------------------------------------

_para_vf_sa_mlp_dict = {
	'class': StateActionValueFunction,
	'args': {
		'state_dim': {'type': 'ref', 'key': 'state_dim', 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'non_linearity': {'type': 'str', 'default': 'relu'},
		'hidden_layers': {'type': 'int', 'default': 1},
		'hidden_dim': {'type': 'int', 'default': 20}
	}
}

_para_value_fun_sa_dict = {
	'default': _para_vf_sa_mlp_dict,
	'mlp': _para_vf_sa_mlp_dict

}

_para_vf_s_mlp_dict = {
	'class': StateValueFunction,
	'args': {
		'state_dim': {'type': 'ref', 'key': 'state_dim', 'subpath': ''},

		'non_linearity': {'type': 'str', 'default': 'relu'},
		'hidden_layers': {'type': 'int', 'default': 1},
		'hidden_dim': {'type': 'int', 'default': 20}
	}
}

_para_value_fun_s_dict = {
	'default': _para_vf_s_mlp_dict,
	'mlp'    : _para_vf_s_mlp_dict
}


# Agents
#--------------------------------------------------------------------------------

_para_ag_reinforce_dict = {
	'class': REINFORCE,
	'args' : OrderedDict({


		'policy'    : {'type': 'obj' , 'dict': _para_policies_reinforce_dict},
		'action_fun': {'type': 'dict', 'dict': _para_action_fun_dict },

		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'gamma'   : {'type': 'float', 'default': 0.99},

		'baseline'     : {'type': 'bool', 'default': True},
		'baseline_fun' : {'type': 'obj' , 'dict': _para_value_fun_s_dict},

		'lr'   : {'type': 'float', 'default': 1e-4},
		'bl_lr': {'type': 'float', 'default': 1e-4}
	})
}

_para_ag_dqn_dict = {
	'class': DQN,
	'args' : OrderedDict({
		'action_fun': {'type': 'dict', 'dict': _para_action_fun_dict},

		'q'       : {'type': 'obj', 'dict': _para_value_fun_sa_dict},
		'q_target': {'type': 'obj', 'dict': _para_value_fun_sa_dict},

		'policy': {'type': 'obj', 'dict': _para_policies_dqn_dict},

		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'subpath': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'subpath': ''},

		'gamma': {'type': 'float', 'default': 0.99},
		'doubleQ': {'type': 'bool', 'default': True},

		'batch_size' : {'type': 'int', 'default': 64},

		'lr': {'type': 'float', 'default': 1e-4},
	})
}

_para_agents_dict = {
	'default'  : _para_ag_reinforce_dict,
	'reinforce': _para_ag_reinforce_dict,
	'dqn'      : _para_ag_dqn_dict
}


# Parameters
#--------------------------------------------------------------------------------
_para_global_param_dict = {
	'model_dir': {'type': 'str', 'default': '../save/models'},
	'plt_dir'  : {'type': 'str', 'default': '../save/plots'},
	'data_dir' : {'type': 'str', 'default': '../save/stats'},

	'show': {'type': 'bool', 'default': False}
}

_para_param_dict = OrderedDict({

	'runs'      : {'type': 'int', 'default': 5},
	'episodes'  : {'type': 'int', 'default': 3000},
	'time_steps': {'type': 'int', 'default': 500},

	'test_episodes': {'type': 'int', 'default': 5},

	'state_dim' : {'type': 'int', 'default': 4},
	'action_dim': {'type': 'int', 'default': 1},

	'initial_state'  : {'type': 'dict', 'dict': _para_init_states_dict},
	'initial_noise'  : {'type': 'dict', 'dict': _para_init_noises_dict},

	'reward_function': {'type': 'obj', 'dict': _para_reward_func_dict},

	'agent' : {'type': 'obj', 'dict': _para_agents_dict}
})



#================================================================================
#
# Recursive Argument parsing function
#
#================================================================================

def _para_parse_arg(key, args, dict, refs):

	type = dict['type']
	parser_type_subdict = _arg_types[type]
	parser_func = parser_type_subdict['func']

	if 'args' in _arg_types[type]:
		parser_args_key = parser_type_subdict['args']
		if parser_args_key in dict:
			parser_dict = dict[parser_args_key]
		else:
			parser_dict = dict
	else:
		parser_dict = dict

	return parser_func(key, args, parser_dict, refs)

def _para_parse_args(key, args, dict, refs):
	parsed_args = {}

	if dict is None:
		return None

	for prop_key, properties in dict.items():

		parsed_args[prop_key] = _para_parse_arg(prop_key, args, properties, refs)

	return parsed_args


#================================================================================
#
# Helper functions
#
#================================================================================

def _para_get_param(dict, key, default=''):
	key = key.lower()
	if key in dict:
		return dict[key]
	else:
		return default

def _para_get_subdict(dict, key):
	return _para_get_param(dict, key, {})


#================================================================================
#
# Main function to load parameters from a file
#
#================================================================================

def load_parameters(file):

	with open(file, 'rb') as f:
		parameters_file = json.load(f)

	parameters_dict = {}
	refs = {}
	experiments = []

	# Add global parameters to dictionary
	parameters_dict['global_params'] = _para_parse_args(None, _para_get_subdict(parameters_file, 'global_params'), _para_global_param_dict, refs)

	for e, exp in enumerate(parameters_file['experiments']):

		experiment_dict = {}

		# Add execution flag to dictionary
		if 'exe' in exp:
			experiment_dict['exe'] = exp['exe']
		else:
			experiment_dict['exe'] = True

		# Add Filename and Description to dictionary
		if 'desc' in exp and 'file_name' in exp:
			experiment_dict['desc'] = exp['desc']
			experiment_dict['file_name'] = exp['file_name']
		elif 'desc' in exp:
			experiment_dict['desc'] = exp['desc']
			experiment_dict['file_name'] = exp['desc']
		elif 'filename' in exp:
			experiment_dict['desc'] = exp['file_name']
			experiment_dict['file_name'] = exp['file_name']
		else:
			exp_name = 'exp_{}'.format(e)
			experiment_dict['desc'] = exp_name
			experiment_dict['file_name'] = exp_name

		refs.update(experiment_dict)

		# Add experiment parameters to dictionary
		experiment_dict.update(_para_parse_args(None, _para_get_subdict(exp, 'params'), _para_param_dict, refs))

		# Add dictionary to experiments list
		experiments.append(experiment_dict)

	parameters_dict['experiments'] = experiments

	return parameters_dict