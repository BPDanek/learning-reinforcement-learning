# learning how to work out at the gym: a.03.2[CartPole-v0]
#
#
# ALGORITHM PSEUDOCODE--------------------------------------------------------------------------------------------
#
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


# IMPORTS ---------------------------------------------------------------------------------------------------------

import numpy as np
import gym
import torch.nn as nn
import torch


# HYPERPARAMETERS ------------------------------------------------------------------------------------------------

# the number of memories used per training step, this behaves as another channel to the input data
MEMORIES_PER_TRAINING = 3

# number of episodes over which to perform learning of Q estimator
EPISODES = 5

EXPLORATION_EXPLOITATION_DECAY = 0.04 # tradeoff between exploration and exploitation decay rate
                                      # the constant that will help the RL system shift from the mindset of
                                      # exploring to exploiting

NUMBER_OF_MEMORIES = 100 # number (volume) of "memories," states that are available in action replay

SIZE_OF_MEMORY_CONTENTS = 4 # number of entries per memory in memory replay: (state, action, reward, new_state)

# cart_pos, cart_velocity, pole_angle, pole_velocity, channel(number of memories) = (4, MEMORIES_PER_TRAINING)
# INPUT_DIM = (
#                 (cart_pos, cart_vel, pole_ang, pole_vel),
#                 (cart_pos, cart_vel, pole_ang, pole_vel),
#                 (cart_pos, cart_vel, pole_ang, pole_vel),
#                 (cart_pos, cart_vel, pole_ang, pole_vel)
#             )

INPUT_DIM_TENSOR = torch.zeros([MEMORIES_PER_TRAINING, SIZE_OF_MEMORY_CONTENTS], dtype=torch.int32)

# print(INPUT_DIM_TENSOR)
# print(INPUT_DIM_TENSOR.size())


# NEURAL NETWORK (CLASS) -----------------------------------------------------------------------------------------

# implementation roughly taken form tutorial:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class DQNetwork(nn.Module):

    def __init__(self): # constructor
        super(DQNetwork, self).__init__() # inherits properties from parent class

        # TODO: run some kind of proof of concept on these--they are not* robust
        # nn layers:
        # since there is no image data, only information based on the cart
        # position, cart velocity, pole angle, pole velocity, we will use linear (FC) layers instead of Conv. layers
        # the architecture needs refinement for sure
        #
        # https://pytorch.org/docs/stable/nn.html#linear

        # done lose track:
        # quality of state, action pair = Q(S,A)
        # the Q(.,.) function is what needs to be approximated with our NN

        self.fully_connected1 = nn.Linear((MEMORIES_PER_TRAINING * SIZE_OF_MEMORY_CONTENTS), 9)
        self.fully_connected2 = nn.Linear(9, 6)
        self.fully_connected3 = nn.Linear(6, MEMORIES_PER_TRAINING)
        # the output: memories_per_training represent the 4 action values derived from this NN approximation

    def forward(self, input_data):
        out = self.fully_connected1(input_data)
        out = self.fully_connected2(out)
        out = self.fully_connected3(out)
        return out


dummy = (2, 2, 2, 2)

# initialize memory replay with size N (NUMBER_OF_MEMORIES)
replay_memory = np.zeros((NUMBER_OF_MEMORIES, SIZE_OF_MEMORY_CONTENTS))
memory_index = 0 # counts index of replay memory

print(replay_memory[0])
replay_memory[0] = dummy
print(replay_memory[0])

net = DQNetwork()
env = gym.make('CartPole-v0')  # CartPole simulation environment from gym


# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]


for episode in range(EPISODES):

    done = False

    observation = env.reset()
    replay_memory[memory_index] = observation

    # while not done:
    #     # select random value



print(net)



