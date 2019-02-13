# learning how to work out at the gym: a.04.0[CartPole-v0]

# IMPORTS ---------------------------------------------------------------------------------------------------------

import numpy as np
import gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------

# number of episodes over which to perform learning of Q estimator
EPISODES = 1


# NEURAL NETWORK (CLASS) -----------------------------------------------------------------------------------------

class DQNetwork(nn.Module):

    # HYPERPARAMETERS

    # don't lose track:
    # quality of state, action pair = Q(S,A)
    # the Q(.,.) function is what needs to be approximated with our NN

    def __init__(self): # constructor
        super(DQNetwork, self).__init__() # inherits properties from parent class

        # nn layers:
        # since there is no image data, only information based on the cart
        # position, cart velocity, pole angle, pole velocity, we will use linear (FC) layers instead of Conv. layers
        # the architecture needs refinement for sure
        #
        # https://pytorch.org/docs/stable/nn.html#linear

        # input shape
        self.fully_connected1 = nn.Linear(#TODO: )
        self.fully_connected2 = nn.Linear(#TODO: )
        self.fully_connected3 = nn.Linear(#TODO: )
        # the output: memories_per_training represent the 4 action values derived from this NN approximation

    def forward(self, input_data):
        out = self.fully_connected1(input_data)
        out = self.fully_connected2(out)
        out = self.fully_connected3(out)
        return out


# Algorithm Implementation ----------------------------------------------------------------------------------------

env = gym.make('CartPole-v0')  # CartPole simulation environment from gym
# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]

net = DQNetwork()

# return argmax (action) given state from Neural Network
# neural network usage derived roughly from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
def exploit_action(state):
    return net(state)



def run_training_operation():

    for episode in range(EPISODES):

        # used in print statements and debugging, should be incremented within while loop
        iteration = 0-1 # subtract 1 since we update iteration immediately within loop
        iterator_plus = 0 # this itorator is one ahead of the origional, needed for pyplot to resolve error:
                          # ValueError: x and y must be the same size where y (iterator) = (2,) and x = (3,)


        # determines when to break out of while loop; triggered by episode limit, or by gym (2nd parameter in step return)
        done = False

        # some parameters:


        # resetting environment returns initial observation
        observation = env.reset()

        # primary algorithm iteration within episode; reset each episode
        # select action, learn something
        #   action determined by exploration-exploitation tradeoff
        #   learning done by NN instance, sample to learn comes from memory bank
        while not done:

            # update some the iterators, as that's their purpose
            iteration += 1; iterator_plus += 1

            # exploration-exploitation tradeoff epsilon used to decide when to exploit and when to explore
            epsilon_rand = np.random.rand()

            # exploration decline is defined as such (by myself, may need to be optimized)
            current_explore_limit = 1/episode

            if epsilon_rand <= current_explore_limit:
                # exploit action
            else:
                # explroe aciton


            # TODO: remove this once building the code is done
            if iteration == 20:
                done = True




# TEST SPACE


# print(range(env.action_space))
# print(net)
net_parameters = list(net.parameters())

print("length of net parameters {}".format(len(net_parameters)))

for i in range(len(net_parameters)):
    print("policy_param @ {}:\t{}".format(i, net_parameters[i].size()))  # shape: number of features in layer, input size into layer

input = torch.randn(4)
print("torch.random input: {}".format(input))

# TODO: make output an action to take
output = net(input)

print("nn output given random input: {}".format(output))

run_training_operation()


