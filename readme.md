# Reinforcement Learning in OpenAI's CartPole environment
## Final Project for the Reinforcement Learning Lecture
#### Arce y de la Borbolla, José <br/> Sälinger, Andreas


This repository contains the code for training, testing and evaluating Reinforcement Learning agents in a custom version
of OpenAI's CartPole environment with continuous action-space, overloaded reward-function and episode-termination conditions.

Currently, the following Reinforcement Learning algorithms have been developed:
* DQN with optional Double-Q and Replay Buffer
* REINFORCE with Gaussian, Beta and MLP policies
* TD3 with MLP Actor and Critics


## Directories:
* **agents** : contains the executable scripts for setting-up, training and testing agents.
* **lib**    : classes, objects and methods, including the main algorithms
* **scripts**: helper scripts for testing and evaluating the agents
* **save**: saved agent models, statistics, output logs, plots and TensorBoard graphs.
* **slides**: Presentation slides with the results
