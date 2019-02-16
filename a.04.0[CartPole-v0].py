# learning how to work out at the gym: a.04.0[CartPole-v0]

# IMPORTS ---------------------------------------------------------------------------------------------------------

import numpy as np
import gym
import torch.nn as nn
from torch import optim as optim
import torch

import matplotlib.pyplot as plt

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------
LEARNING_RATE = 0.007

# number of episodes over which to perform learning of Q estimator
EPISODES = 50
MAX_TIMESTEP = 70


# NEURAL NETWORK (CLASS) -----------------------------------------------------------------------------------------

"""
Neural network estimates Q-value
NN estimates Q(S,A)
S = state = observation @t (4,)
A = action = one of possible actions @state (1,) [0 or 1 --> left or right]
Since this problem only has 2 actions at each time step (left or right) we will make the input into the neural network
a combination of S A.

Input: S.append(A) = (4,) + (1,) = (5,); np.zeros((5,)) = [0,0,0,0,0]
(first 4 digits are observation, last digit is action Q(S,A))

Output: floating point value denoting value of current state-action pair
"""
class DQNetwork(nn.Module):

    def __init__(self):

        super(DQNetwork, self).__init__() # inherits properties from parent class

        # nn layers:
        # since there is no image data, only information based on the cart
        # position, cart velocity, pole angle, pole velocity, we will use linear (FC) layers instead of Conv. layers
        # the (todo: )architecture needs refinement for sure
        #
        # https://pytorch.org/docs/stable/nn.html#linear

        self.fully_connected1 = nn.Linear(in_features=5, out_features=4)
        self.fully_connected2 = nn.Linear(in_features=4, out_features=3)
        self.fully_connected3 = nn.Linear(in_features=3, out_features=2)
        self.fully_connected4 = nn.Linear(in_features=2, out_features=1)


    # todo: identify if these activations are reasonable
    def forward(self, input_data):
        out = nn.functional.relu(self.fully_connected1(input_data))
        out = nn.functional.relu(self.fully_connected2(out))
        out = nn.functional.relu(self.fully_connected3(out))
        out = self.fully_connected4(out)
        return out


# functions -------------------------------------------------------------------------------------------------------
"""
The following function performs a forward pass in the DQN (neural network) object.
It requires that we: asses the quality of each state-action pair, and return the highest valued one
This task requires that we: 

append current observation to action_left
append current observation to action_right
pass both through dqn
compare them
identify highest value one
return action (in the form of env.action_space [either 0 or 1])

"""
def exploit_action(observation, DQN, env):
    # this env-specific action set may be backwards, but these are the discrete actions within the action space of cartpole
    ACTION_LEFT = [0]
    ACTION_RIGHT = [1]

    # since we're passing parameters into the nn made in the torch framework we cast them to the torch tensor dtype
    # inputs to neural network
    combined_in_L = torch.FloatTensor(np.append(observation,  np.asarray(ACTION_LEFT)))
    combined_in_R = torch.FloatTensor(np.append(observation, np.asarray(ACTION_RIGHT)))

    Q_value_L = DQN(combined_in_L)
    Q_value_R = DQN(combined_in_R)

    if Q_value_L > Q_value_R:
        return 0
    elif Q_value_L < Q_value_R:
        return 1
    else:
        print("\nQ-Values all equal, sampling action_space")
        return env.action_space.sample()

"""
This method returns the numeric value of the optimal Q value given a state, network object that is trained.
Since this method may have no optimal Q value, we need the env so that we can sample the action space randomly
"""
def Q_numeric_val(observation, DQN, env):
    # compute the ideal action at observation, per observation, and add the two into a tensor
    optimal_action_at_state = exploit_action(observation, DQN, env)

    input = torch.FloatTensor(np.append(observation, optimal_action_at_state))

    # pass the tensor through the network, and return the numeric value of that tensor
    return DQN(input)

# Algorithm Implementation ----------------------------------------------------------------------------------------


env = gym.make('CartPole-v0')  # CartPole simulation environment from gym
# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]

# make a DQN object
DQN = DQNetwork()

loss_accumulator = np.zeros((EPISODES, MAX_TIMESTEP))

def run_training_operation():

    # forward declaration to define scope of variable used in plotting later on, and to describe timesteps within
    # an episode
    iteration = 0
    for episode in range(EPISODES):

        print("\nEpisode: ", episode)

        # used in print statements and debugging, should be incremented within while loop
        iteration = -1               # subtract 1 since we update iteration immediately within loop

        done = False                # flag that breaks out of episode; changed when env returns "done"

        observation = env.reset()   # resetting environment returns initial observation

        # primary algorithm iteration within episode; reset each episode
        # select action, learn something
        #   action determined by exploration-exploitation trade-off
        #   learning done by NN instance, sample to learn comes from direct memory
        while not done:

            # update some the iterators, as that's their purpose
            iteration += 1

            # exploration-exploitation tradeoff epsilon used to decide when to exploit and when to explore
            epsilon_rand = np.random.rand()

            # exploration decline is defined as such; seen in sutton/barto's SARSA
            current_explore_limit = 1/(episode + 1)

            # explore limit decreases with time. epsilon greater than limit, explore; else explopit
            if epsilon_rand <= current_explore_limit:
                action = exploit_action(observation, DQN, env)
            else:
                action = env.action_space.sample()  # take random (exploratory) action

            # take step, observe reward, check for done
            # this step updates: state = state++
            next_observation, reward, done, _ = env.step(action=action)

            # TODO: remove this once building the code is done
            # break at a time step limit
            if iteration == MAX_TIMESTEP:
                done = True


            # learning step

            # target Q_val = reward + Q(S',A)
            Y = reward + Q_numeric_val(next_observation, DQN, env)

            # real Q_val = Q(S,A)
            Q = Q_numeric_val(observation, DQN, env)

            # use mse loss for computing difference between target Q and actual Q
            loss_fn = nn.MSELoss()
            loss = loss_fn(Y, Q)

            loss_accumulator[episode][iteration-1] = loss

            print(observation)
            print(next_observation)

            print("Loss at timestep {}:     {}".format(iteration, loss))
            print("Y = reward + Q(S',A'):   {}".format(Y))
            print("Q = Q(S,A)):             {}".format(Q))

            # use optimizer with stochastic gradient descent
            optimizer = optim.SGD(DQN.parameters(), lr=LEARNING_RATE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    x = range(MAX_TIMESTEP)
    y_0 = loss_accumulator[30]
    y_1 = loss_accumulator[45]
    y_2 = loss_accumulator[49]

    plt.plot(x, y_0)
    plt.plot(x, y_1)
    plt.plot(x, y_2)

    plt.xlabel('timesteps')
    plt.ylabel('loss_accumulator object')
    plt.legend(['episode n', 'episode n+1', 'episode n+2'])
    plt.show()

run_training_operation()

# TEST SPACE
#
# action = env.action_space.sample()
# print("action: ", action)
#
# net = DQNetwork()
# print("net:", net)
# print("zeros in observation space", np.zeros(np.shape(env.observation_space.high)))
# observation = env.reset()
# observation = torch.FloatTensor(observation)
# print("observation from env as tensor: ", (observation))
# print("observation shape", np.shape(env.observation_space.high))
# print("observation tensor shape: ", np.shape(observation))
#
#
# input = (torch.FloatTensor([0]) + torch.FloatTensor(observation))
# print("torch.random input: {}".format(input))
#
# # TODO: make output an action to take
# output = net(input)
#
# print("nn output given random input: {}".format(output))

#run_training_operation()


# todo: eliminate env from exploit action so that implementation is cleaner (maybe faster cuz no calls to other methods)
# QUESTIONS FOR MAX:
# how do I encode discount, and wtf is it

# todo: how the fuck do i make my model better? I must be missing something critical here...