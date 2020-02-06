import numpy as np

init_state_up = np.zeros(4)

initial_states = {
	'no': None, 'none': None,
	'up': init_state_up
}

# Initial Noise
init_noise_360 = np.array([0.5, 0.5, np.pi, 0.5])

initial_noises = {
	'no': None, 'none': None,
	'360': init_noise_360
}