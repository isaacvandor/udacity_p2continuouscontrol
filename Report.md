# Continuous Control Deep Deterministic Policy Gradients Project Report

## Overview

This report describes my solution to the Reacher environment (option one) as part of 
the Continuous Control project from Udacity Deep Reinforcement Learning Nanodegree.
The implementation I developed solves the environment in 343 episodes with an average score of +30 over 100 consecutive episodes.

This implementation is heavily inspired by the "Continuous Control with Deep Reinforcement Learning" (Lillicrap, Timothy P., et al) paper as well as:

- [Udacity Deep Reinforcement Learning Nanodegree repository](https://github.com/udacity/deep-reinforcement-learning) 
- [DeepRL repository](https://github.com/ShangtongZhang/DeepRL)

## Learning Algorithm

The DDPG algorithm is implemented in the [ddpg_main.py](ddpg_main.py) file and utilizes an Actor and Critic class model. 
The actor learns to predict an action vector while the critic learns Q values for state-action pairs. Online and target models are used for the Actor and critic in order to avoid overestimation of Q-values. Additionally, this implementation utilizes experience replay (see class for implementation) to train on randomly sampled batches of uncorrelated experiences.

The DDPG agent is implemented in [ddpg_agent.py](ddpg_agent.py) and the Actor and Critic are implemented in [ddpg_model.py](ddpg_model.py). 

The agent works as follows:

First, actions are generated for the current state with the online actor model
Then, noise is added using the Ornstein-Uhlenbeck process
Learning is then done in the following way:

1. Experiences are sampled in batches from the replay buffer
2. The online critic model is updated
3. The online actor model is updated
4. The target critic and actor models are updated through soft update and using the parameter vals from the online models

Training takes place in [ddpg_main.py](ddpg_main.py), specifically in the `run` function, which works like so:
1. The environment is observed at every timestep
2. The agent (loaded from [ddpg_agent.py](ddpg_agent.py)) picks an action
3. The next state, received reward, and whether the episode is completed are observed in the environment
4. This SARSA experience is then added to the replay buffer to be sampled later
5. The agent then uses these experiences to learn and improve aka training. See the implementation for more details on training.

## Network architecture

DDPG uses 2 network architectures: one for the actor and one for the critic.

Actor network architecture:
1. State input (33 units)
2. Hidden layer (256 units) with ReLU activation and batch normalization
3. Hidden layer (128 units) with ReLU activation and batch normalization
4. Action output (4 units) with tanh activation

Critic network architecture:
1. State input (33 units)
2. Hidden layer (256 nodes) with ReLU activation and batch normalization
3. Action input (4 units)
4. Hidden layer with inputs from layers 2 and 3 (128 nodes) with ReLU activation and batch normalization
5. Q-value output (1 node)

## Config Parameters

Initial param values and network architectures came from the original paper and provided reference implementations. Params were then tweaked, especially the sigma noise value, learning rates, and the network architecture nodes. Param values can be found in [ddpg_main.py](ddpg_main.py) in the `main` function.

## Results
The agent solved the environment by scoring 30+ over 100 consecutive episodes (the definition of success provided by Udacity) in 343 episodes. 

Here is a plot of the scores for each episode:

![scores_plot](final_scores.png)

## Considerations for future work
1. Tweaking some of the parameters to find optimal values
2. Tweaking the network architectures to more successfully solve the environment
3. I had many...many...many issues with the Jupyter workspace and I would love to work on this in the future. So many issues. I spent a considerable amount of time debugging the workspace before switching over to a pure python implementation.

