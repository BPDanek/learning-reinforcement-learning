# learning how to work out at the gym: a.03.2[CartPole-v0]
#
#
# ALGORITHM PSEUDOCODE--------------------------------------------------------------------------------------------
#
# DQN Algorith`m Overview:
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


# things to change:
# explore/exploit decay rate should probably have some scientific backing (i feel like it should be more gradual)
# neural network architecture (layers, number of nodes, activations)
# imporove means of plotting pyplot stuff (excessive variables right now, really complicated)
# explain the "exploration limit" better
# explore action function paired to exploit action function

# IMPORTS ---------------------------------------------------------------------------------------------------------

import numpy as np
import gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------

# the number of memories used per training step, this behaves as another channel to the input data
MEMORIES_PER_TRAINING = 3

# number of episodes over which to perform learning of Q estimator
EPISODES = 1

NUMBER_OF_MEMORIES = 1000 # number (volume) of "memories," states that are available in action replay

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


# Algorithm Implementation ----------------------------------------------------------------------------------------


# initialize memory replay with size N (NUMBER_OF_MEMORIES)
replay_memory = np.zeros((NUMBER_OF_MEMORIES, SIZE_OF_MEMORY_CONTENTS))
memory_index = 0 # counts index of replay memory

env = gym.make('CartPole-v0')  # CartPole simulation environment from gym
# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]

policy_net = DQNetwork()
target_net = DQNetwork()




# return argmax (action) given state from Neural Network
# neural network usage derived roughly from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
def exploit_action():




def run_training_operation(verbose=False, show_plot=False):

    for episode in range(EPISODES):

        # used in print statements and debugging, should be incremented within while loop
        iteration = 0-1 # subtract 1 since we update iteration immediately within loop
        iterator_plus = 0 # this itorator is one ahead of the origional, needed for pyplot to resolve error:
                          # ValueError: x and y must be the same size where y (iterator) = (2,) and x = (3,)

        # determines when to break out of while loop; triggered by episode limit, or by gym (2nd parameter in step return)
        done = False

        # some parameters:
        # exploration-exploitation cutoffs, epsilon below explore, explore. Epsilon above explore, exploit
        explore_var_start = 0  # a value that will be iteratively updated, and used to derive our exploration limit
        explore_min = 0.01
        explore_decay = 0.01  # explore value will decay within the while loop at this fixed rate
        explore_decay_step = 0  # a component of the function that adjusts the bar used to determine this tradeoff

        # resetting environment returns initial observation
        observation = env.reset()

        # place initial observation into memory bank at current empty index
        replay_memory[memory_index] = observation

        # some pyplot utility variables:
        # these two variables are intended to be plotted vs. iteration (show_plot == True)
        current_explore_limit_BUFFER = []
        explore_var_start_BUFFER = []


        # primary algorithm iteration within episode; reset each episode
        # select action, learn something
        #   action determined by exploration-exploitation tradeoff
        #   learning done by NN instance, sample to learn comes from memory bank
        while not done:

            # update some iterators, as that's their purpose
            iteration += 1; iterator_plus += 1

            # exploration-exploitation tradeoff epsilon used to decide when to exploit and when to explore
            epsilon_rand = np.random.rand()

            # exploration decline is defined as such (by myself, may need to be optimized)
            current_explore_limit = 1 - (explore_var_start ** 2)

            # update a log of a changing variable for pyplot; this variable changes per iteration of while loop
            current_explore_limit_BUFFER.append(current_explore_limit)

            # increment before we update explore_var_start
            explore_decay_step += 1

            # calculating decay of exploration probability:
            # we don't want to decay below zero, always retain some small* chance of
            # chose (3/8) because this roughly translates to "below 13%" (normal dist)
            if not current_explore_limit < (3/8):
                # need to update explore_start rate so that it's a little bigger each time
                explore_var_start = explore_var_start + explore_decay*explore_decay_step

            # update a log of a changing variable for pyplot; this variable changes per iteration of while loop
            explore_var_start_BUFFER.append(explore_var_start)

            # TODO: remove this once building the code is done
            if iteration == 20:
                done = True


            # select action to take
            if current_explore_limit > epsilon_rand:
                action = env.action_space.sample() # explore state space for new income
            else:
                action = exploit_action(observation) # exploit state space for best action




            # ------ (start) some loud output statements ------
            # verbosity flag triggers some information about the current training loop
            if verbose is True:

                if iteration == 0:  # start of while (unique print)
                    print("\n\n----Episode {} initiated----".format(episode))

                print("\n\nIteration {}\n".format(iteration))
                print("\texplore_decay_step: {}".format(explore_decay_step))
                print("\tepsilon_rand: {:.4f}".format(epsilon_rand))
                print("\texplore_var_start: {}".format(explore_var_start))
                print("\tcurrent_explore_limit: {:.4f}".format(current_explore_limit))

                print("\t\tvar: iteration range            {}".format((np.asarray(range(iteration))).shape))
                print("\t\tvar: iteratr_plus range         {}".format((np.asarray(range(iterator_plus))).shape))
                print("\t\tvar: explore_var_start size     {}".format((np.asarray(explore_var_start_BUFFER)).shape))
                print("\t\tvar: current_explore_limit size {}".format((np.asarray(current_explore_limit_BUFFER)).shape))

            # plotting explore_limit_var and explore_limit as a matplotlib.pyplot versus iterations to explore the
            # details of exploration-exploitation tradeoff
            # this functionality occurs only after an episode is complete, really only useful during debugging
            if show_plot is True and done is True:
                # reminder: explore limit is derived from explore variable
                #plot_iteration = np.linspace(0, iteration[-1])
                plt.scatter(np.asarray(range(iterator_plus)), np.asarray(explore_var_start_BUFFER),
                            label="exploration variable")
                plt.scatter(np.asarray(range(iterator_plus)), np.asarray(current_explore_limit_BUFFER),
                            label="explore limit")

                plt.xlabel("iteration")
                plt.ylabel("exploration variable and limit")
                plt.legend()
                plt.show()
            # ------ (end) some loud output statements ------


            # now that we've made probabilities for our explore/exploit tradeoff let's implement some actions



#  print(range(env.action_space))
#  print(net)
run_training_operation(verbose=True, show_plot=True)


