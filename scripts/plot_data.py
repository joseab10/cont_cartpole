import pickle

from utils import *

with open('../save/data/datadisc_dqn_20200128_080756.pkl', 'rb') as f:
	data = pickle.load(f)

train_length_means = np.mean(data[0]['lengths'], axis=0)
train_length_stdev =  np.std(data[0]['lengths'], axis=0)
train_reward_means = np.mean(data[0]['rewards'], axis=0)
train_reward_stdev =  np.std(data[0]['lengths'], axis=0)
test_length_means = np.mean(data[1]['lengths'], axis=0)
test_length_stdev =  np.std(data[1]['lengths'], axis=0)
test_reward_means = np.mean(data[1]['rewards'], axis=0)
test_reward_stdev =  np.std(data[1]['lengths'], axis=0)

new_data = [{'run': 'train',
						 'reward_means': train_reward_means, 'reward_stdev': train_reward_stdev, 'rewards': data[0]['rewards'],
						 'length_means': train_length_means, 'length_stdev': train_length_stdev, 'lengths': data[0]['lengths']},
			{'run': 'test' ,
						 'reward_means': test_reward_means, 'reward_stdev': test_reward_stdev, 'rewards': data[1]['rewards'],
						 'length_means': test_length_means, 'length_stdev': test_length_stdev, 'lengths': data[1]['lengths']}]

plot_run_stats(new_data, '../save/plots','123', plot_runs=True, noshow=True)