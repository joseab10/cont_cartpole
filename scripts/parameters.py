import numpy as np

from continuous_reinforce import REINFORCE
from deep_q_learning import DQN
from policies import GaussPolicy, BetaPolicy, MlpPolicy, EpsilonGreedyPolicy
from action_functions import ActCont2Cont, ActDisc2Cont
from value_functions import StateActionValueFunction, StateValueFunction
from schedules import sch_no_decay, sch_linear_decay, sch_exp_decay, Schedule
from utils import tn

from continuous_cartpole import angle_normalize

from collections import OrderedDict
import json


# ================================================================================
#
# "Public" Methods
#
# ================================================================================

def parse_reward_function(key, refs):
	return _para_parse_obj(key, {}, _para_reward_func_dict, refs, add_ref=False)


def parse_init_state(key):
	refs = {}
	return _para_parse_primitive(key, {}, _para_init_states_dict, refs, add_ref=False)


def parse_init_noise(key):
	refs = {}
	return _para_parse_primitive(key, {}, _para_init_noises_dict, refs, add_ref=False)


# ================================================================================
#
# Type parsing functions
#
# ================================================================================

def _para_add_ref(refs, key, value, add_ref=True):
	if add_ref:
		refs[key] = value


def _para_parse_primitive(key, args, dic, refs, add_ref=True):
	if key in args:
		value = args[key]
	else:
		value = dic['default']

	_para_add_ref(refs, key, value, add_ref=add_ref)

	return value


def _para_parse_dict(key, args, dic, refs, add_ref=True):

	value = dic['default']

	if key in args:
		if args[key] in dic:
			value = dic[args[key]]

	_para_add_ref(refs, key, value, add_ref=add_ref)

	return value


def _para_parse_obj(key, args, dic, refs, add_ref=True):

	args_sub_dict = _para_get_sub_dict(args, key)
	args_class_name = _para_get_param(args_sub_dict, 'class')

	obj_class_dict = _para_parse_primitive(args_class_name, dic, dic, refs, add_ref=False)
	obj_class = _para_get_param(obj_class_dict, 'class')

	if obj_class is None:
		return None

	constructor_args = _para_parse_args(None, _para_get_sub_dict(args_sub_dict, 'args'), obj_class_dict['args'], refs)

	obj = obj_class(**constructor_args)

	_para_add_ref(refs, key, obj, add_ref=add_ref)

	return obj


def _para_parse_ref(key, args, dic, refs):

	ref_key       = _para_get_param(dic, 'key')
	ref_object    = _para_get_param(refs, ref_key, None)
	ref_sub_path  = _para_get_param(dic, 'sub_path')

	if ref_sub_path != '':
		ref_sub_path = ref_sub_path.split('.')
	else:
		ref_sub_path = []

	while ref_object is not None and len(ref_sub_path) > 0:
		if hasattr(ref_object, ref_sub_path[0]):
			ref_object = getattr(ref_object, ref_sub_path[0])
			ref_sub_path = ref_sub_path[1:]
		else:
			ref_object = None

	return ref_object


# ================================================================================
#
# Parameter Dictionaries, objects, functions default values and
# other data structures
#
# ================================================================================

# Argument Parsing
# --------------------------------------------------------------------------------
_arg_func_obj  = {'func': _para_parse_obj, 'args': 'dict'}
_arg_func_dict = {'func': _para_parse_dict, 'args': 'dict'}
_arg_func_prim = {'func': _para_parse_primitive}
_arg_func_ref  = {'func': _para_parse_ref, 'args': 'dict'}

_arg_types = {
	'obj': _arg_func_obj,
	'ref': _arg_func_ref,
	'dict': _arg_func_dict,
	'int': _arg_func_prim, 'float': _arg_func_prim, 'str': _arg_func_prim, 'bool': _arg_func_prim,
}

# Initial State
# --------------------------------------------------------------------------------
init_state_up = np.zeros(4)

_para_init_states_dict = {
	'0': None, 'no': None, 'none': None, 'default': None,
	'1': init_state_up, 'up': init_state_up, 'upright': init_state_up
}


# Initial Noise
# --------------------------------------------------------------------------------
init_noise_360 = np.array([0.5, 0.5, np.pi, 0.5])

_para_init_noises_dict = {
	'0': 0, 'det': 0, 'deterministic': 0, 'no': 0, 'none': 0,
	'1': None, 'default': None,
	'2': init_noise_360, '360_deg': init_noise_360, '360': init_noise_360
}


# Reward Functions
# --------------------------------------------------------------------------------
def informative_reward_generator(time_steps):

	def informative_reward(cart_pole):

		cos_pow = 3
		max_pts = 100

		if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
			return -max_pts
		else:
			return (np.cos(cart_pole.state[2])**cos_pow)*(max_pts/(2 * time_steps))

	return informative_reward


def pos_sparse_rf():
	def positive_rf(cart_pole):
		return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 0

	return positive_rf


_para_none_obj_dict = {
	'class': None,
	'args' : {}
}
_para_rf_informative_dict = {
	'class': informative_reward_generator,
	'args' : {
		'time_steps': {'type': 'ref', 'key': 'time_steps', 'sub_path': ''}
	}
}

_para_rf_pos_sparse_dict = {
	'class': pos_sparse_rf,
	'args' : {}
}

_para_reward_func_dict = {
	'0': _para_none_obj_dict, 'no': _para_none_obj_dict, 'default': _para_none_obj_dict,
	'1': _para_rf_informative_dict, 'info': _para_rf_informative_dict, 'informative': _para_rf_informative_dict,
	'pos_sparse': _para_rf_pos_sparse_dict
}


# Action Functions
# --------------------------------------------------------------------------------
def identity(a):
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
act_identity  = ActCont2Cont(identity, identity)

_para_action_fun_dict = {
	'default': act_identity,
	'clip' : act_clipcont ,
	'2' : act_disc2cont, 'disc2': act_disc2cont,
	'3' : act_disc3cont, 'disc3': act_disc3cont,
	'4' : act_disc4cont, 'disc4': act_disc4cont,
	'5' : act_disc5cont, 'disc5': act_disc5cont,
	'8' : act_disc8cont, 'disc8': act_disc8cont,
	'9' : act_disc9cont, 'disc9': act_disc9cont
}

# Policies
# --------------------------------------------------------------------------------
_para_sch_decay_fun = {
	'none' : sch_no_decay, 'constant': sch_no_decay, 'default': sch_no_decay,
	'linear': sch_linear_decay,
	'exp' : sch_exp_decay, 'exponential': sch_exp_decay
}

_para_multi_sch_dict = {
	'class': Schedule,
	'args' : {
		'x0' : {'type': 'int'  , 'default': 0},
		'x1' : {'type': 'int'  , 'default': 500},
		'y0' : {'type': 'float', 'default': 0.2},
		'y1' : {'type': 'float', 'default': 0.2},

		'schedule_function' : {'type': 'dict', 'dict': _para_sch_decay_fun},

		'cosine_annealing' : {'type': 'bool', 'default': False},
		'annealing_cycles' : {'type': 'int' , 'default': 5}
	}
}

_para_sch_dict = {
	'default': _para_multi_sch_dict,
	'schedule': _para_multi_sch_dict
}

_para_pi_gaussian_dict = {
	'class': GaussPolicy,
	'args' : {
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

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
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

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
		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

		'act_min'    : {'type': 'float', 'default': -1},
		'act_max'    : {'type': 'float', 'default': 1},
		'act_samples': {'type': 'int', 'default': 100},

		'non_linearity'        : {'type': 'str', 'default': 'relu'},
		'hidden_layers'        : {'type': 'int', 'default': 1},
		'hidden_dim'           : {'type': 'int', 'default': 20},
		'output_non_linearity' : {'type': 'str', 'default': 'sigmoid'}
	}
}

_para_pi_e_greedy_dict = {
	'class': EpsilonGreedyPolicy,
	'args' : {
		'schedule': {'type': 'obj', 'dict': _para_sch_dict},

		'value_function': {'type': 'ref', 'key': 'q', 'sub_path': ''}
	}
}

_para_policies_reinforce_dict = {
	'default' : _para_pi_beta_dict,
	'gaussian': _para_pi_gaussian_dict,
	'beta'    : _para_pi_beta_dict,
	'mlp'     : _para_pi_mlp_dict
}

_para_policies_dqn_dict = {
	'default': _para_pi_e_greedy_dict,
	'epsilon_greedy': _para_pi_e_greedy_dict,
	'egreedy': _para_pi_e_greedy_dict
}


# Value Functions
# --------------------------------------------------------------------------------

_para_vf_sa_mlp_dict = {
	'class': StateActionValueFunction,
	'args': {
		'state_dim': {'type': 'ref', 'key': 'state_dim', 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

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
		'state_dim': {'type': 'ref', 'key': 'state_dim', 'sub_path': ''},

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
# --------------------------------------------------------------------------------

_para_ag_reinforce_dict = {
	'class': REINFORCE,
	'args' : OrderedDict({


		'policy'    : {'type': 'obj' , 'dict': _para_policies_reinforce_dict},
		'action_fun': {'type': 'dict', 'dict': _para_action_fun_dict},

		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

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

		'state_dim' : {'type': 'ref', 'key': 'state_dim' , 'sub_path': ''},
		'action_dim': {'type': 'ref', 'key': 'action_dim', 'sub_path': ''},

		'gamma': {'type': 'float', 'default': 0.99},
		'double_q': {'type': 'bool', 'default': True},

		'max_buffer_size': {'type': 'int', 'default': 1e6},
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
# --------------------------------------------------------------------------------
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


# ================================================================================
#
# Recursive Argument parsing function
#
# ================================================================================
def _para_parse_arg(key, args, dic, refs):

	arg_type = dic['type']
	parser_type_sub_dict = _arg_types[arg_type]
	parser_func = parser_type_sub_dict['func']

	if 'args' in _arg_types[arg_type]:
		parser_args_key = parser_type_sub_dict['args']
		if parser_args_key in dic:
			parser_dict = dic[parser_args_key]
		else:
			parser_dict = dic
	else:
		parser_dict = dic

	return parser_func(key, args, parser_dict, refs)


def _para_parse_args(key, args, dic, refs):
	parsed_args = {}

	if dic is None:
		return None

	for prop_key, properties in dic.items():

		parsed_args[prop_key] = _para_parse_arg(prop_key, args, properties, refs)

	return parsed_args


# ================================================================================
#
# Helper functions
#
# ================================================================================
def _para_get_param(dic, key, default=''):
	key = key.lower()
	if key in dic:
		return dic[key]
	else:
		return default


def _para_get_sub_dict(dic, key):
	return _para_get_param(dic, key, {})


# ================================================================================
#
# Main function to load parameters from a file
#
# ================================================================================
def load_parameters(file):

	with open(file, 'rb') as f:
		parameters_file = json.load(f)

	parameters_dict = {}
	refs = {}
	experiments = []

	# Add global parameters to dictionary
	parameters_dict['global_params'] = _para_parse_args(None, _para_get_sub_dict(parameters_file, 'global_params'),
														_para_global_param_dict, refs)

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
		experiment_dict.update(_para_parse_args(None, _para_get_sub_dict(exp, 'params'), _para_param_dict, refs))

		# Add dictionary to experiments list
		experiments.append(experiment_dict)

	parameters_dict['experiments'] = experiments

	return parameters_dict
