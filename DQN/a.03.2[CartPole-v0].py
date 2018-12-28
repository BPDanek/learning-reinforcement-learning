# DQN Algorithm Overview:
# initialize replay memory M
# initialize action-value function Q with weights (a NN)
#
# FOR episode in number of episodes with memory M:
#   initialize sequence s_1 (initial state), preprocess initial sequence (N/A)
#
#   FOR t = 0 to t = end [length of episode]:
#       with exploration exploitation tradeoff epsilon (e) probability
#       IF explore:
#           set random action
#       IF exploit:
#           set best action given current NN knowledge
#       execute set action, observe reward and the consecutive state
#       set new state to current state (s = s')
#       store transition (s, a, r, s') in replay memory M
#
#       (learn from minibatch of transitions)
#       sample minibatch of transitions (s, a, r, s') from M
#       ...
#       ...

import numpy as np
import gym
import torch.nn as nn
#import DQNetwork


NUMBER_OF_MEMORIES = 100 # number of "memories," states that are available in action replay
SIZE_OF_MEMORY_CONTENTS = 4 # (state, action, reward, new_state)


# initialize memory replay with size N (NUMBER_OF_MEMORIES)
replay_memory = np.zeros((NUMBER_OF_MEMORIES, SIZE_OF_MEMORY_CONTENTS))

#dqn = DQN()

env = gym.make('CartPole-v0')

# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]
observation = env.reset()
print(observation)

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print(model)