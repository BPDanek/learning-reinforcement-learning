# learning how to work out at the gym: a.04.1[CartPole-v0]
# todo: Implement experience replay into this bitch
#test

# IMPORTS ---------------------------------------------------------------------------------------------------------
import numpy as np
import gym
import torch.nn as nn
from torch import optim as optim
import torch
import matplotlib.pyplot as plt
import pandas as pd

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------
LEARNING_RATE = 0.01

# number of episodes over which to perform learning of Q estimator
EPISODES = 41
MAX_TIMESTEP = 50

# every __ episodes compute loss, and propogate
NUMBER_OF_MEMORIES = 500

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
def exploit_action(observation, env, DQN):
    # this env-specific action set may be backwards, but these are the discrete actions within the action space of cartpole
    ACTION_LEFT = [0]
    ACTION_RIGHT = [1]

    # since we're passing parameters into the nn made in the torch framework we cast them to the torch tensor dtype
    # inputs to neural network
    combined_in_L = torch.FloatTensor(np.append(observation,  np.asarray(ACTION_LEFT)))
    combined_in_R = torch.FloatTensor(np.append(observation, np.asarray(ACTION_RIGHT)))

    # DQN.zero_grad()

    Q_value_L = DQN(combined_in_L)
    Q_value_R = DQN(combined_in_R)

    print("input L: {}                              Q_left: {}".format(combined_in_L, Q_value_L))
    print("input R: {}                              Q_right: {}".format(combined_in_R, Q_value_R))

    if Q_value_L > Q_value_R:
        print("L")
        return 0
    elif Q_value_L < Q_value_R:
        print("R")
        return 1
    else:
        print("\nQ-Values all equal, sampling action_space")
        return env.action_space.sample()

"""
This method returns the numeric value of the optimal Q value given a state, network object that is trained.
Since this method may have no optimal Q value, we need the env so that we can sample the action space randomly
"""
def Q_numeric_val(observation, action, DQN):
    # since DQN intakes state and action concatenation, we can recreate the correct input to the network, assuming
    # the state of the network is unchanged

    # input is the state_action pair concatenate
    input = torch.FloatTensor(np.append(observation,  action))

    # zero gradient buffers, avoid autograd differentiation via current input. We don't want to overwrite buffers
    # since we need their information (gradients) still
    DQN.zero_grad()

    return DQN(input)


# Algorithm Implementation ---------------------------------------------------------------------------------------

# make a DQN object:
# CartPole simulation environment from gym
env = gym.make('CartPole-v0')
DQN = DQNetwork()


# make structure for visualizing data:
reward_accumulator = np.ndarray((EPISODES, 1))
loss_accumulator = np.ndarray((EPISODES, 1))


def run_training_operation():

    # declare MSE loss function:
    loss_fn = nn.MSELoss()


    # declare experience replay:

    # shape of  data struct within experience replay
    observation_shape = [[NUMBER_OF_MEMORIES], [NUMBER_OF_MEMORIES], [NUMBER_OF_MEMORIES], [NUMBER_OF_MEMORIES]]

    # the following wild declaration of an array creates an array of dictionaries, which holds data from a "memory"
    # which is the necessary data for a single iteration in the experience replay. Inspired by pandas being shit for RL
    experience_replay = [{
        'Y'           : -1,
        'observation0': -1,
        'observation1': -1,
        'observation2': -1,
        'observation3': -1,
        'action'      : -1
    } for k in range(NUMBER_OF_MEMORIES)]

    # the counter used to indicate which memory we are writing when we overwrite our data/fill the memories in training
    experience_counter = 0

    # declaration and initialization of observation data structure. To be used later in program during exp. replay
    observation_exp_rep = [0, 0, 0, 0]

    # episode training loop:
    for episode in range(EPISODES):

        # some formatting
        print("="*45)
        print("\nEpisode: ", episode)


        # declarations/initializations for episodes
        # time-step of current episode
        time_step = 0

        # flag that breaks out of episode; changed when env returns "done"
        done = False

        # resetting environment returns initial observation
        observation = env.reset()

        # these two variables will be used to demonstrate quality of training later
        sum_reward = 0
        sum_loss = 0

        # episode run:
        while done is not True:


            # select action to take:
            # generate random number to be epsilon rand
            epsilon_rand = np.random.rand()

            # set limit for exploration: if above limit, exploit, if below, explore
            # limit declines with time, so being above it becomes more likely
            current_explore_limit = 1/(episode + 1)

            # explore limit is declining
            # if greater than limit, exploit (more likely with time)
            # if less than limit, explore (less likely with time)
            if epsilon_rand >= current_explore_limit:
                action = exploit_action(observation, env, DQN)
            else:
                action = env.action_space.sample()  # take random (exploratory) action


            # interact with env:
            # take step, observe reward, check for done
            # this step updates: state = state++
            next_observation, reward, done, info = env.step(action=action)


            # sum of rewards per episode to be stored in accumulator so we can graph it
            sum_reward = sum_reward + reward


            # learning step:
            # compute corresponding optimal action "next_action" to its state "next_state"
            # these values will be used to calculate target value of NN
            next_action = exploit_action(next_observation, env, DQN)

            # filling exp replay:
            # target Q_val = reward + Q(S',A), will be stored in buffer
            # note: in other RL environments we would have a Y calculation for non-terminal and terminal reward
            #       since there is no Q_numeric if we are *done*
            Y = reward + Q_numeric_val(next_observation, next_action, DQN)

            # fill the wild data structure with info for Y, observation, action
            # Y = reward + Q(s', a') <-- value stored
            # Q = Q(s, a) <-- these state & action are the ones stored
            (experience_replay[experience_counter])['Y']            = Y
            (experience_replay[experience_counter])['observation0'] = observation[0]
            (experience_replay[experience_counter])['observation1'] = observation[1]
            (experience_replay[experience_counter])['observation2'] = observation[2]
            (experience_replay[experience_counter])['observation3'] = observation[3]
            (experience_replay[experience_counter])['action']       = action

            experience_counter += 1


            # learning from experience replay asynchronously:
            # experience replay implementation. Y, state, action should be in replay. Asynchronously learn from
            # instances of transitions. This makes for good learning-why? <--todo
            if experience_counter is not 0:

                # recall a memory:
                # sample micro-experience from experience memory that shows a single state transition
                learning_index = np.random.randint(0, experience_counter)

                # write memory data from data structure
                Y_exp_rep               = (experience_replay[learning_index])['Y']
                observation_exp_rep[0]  = (experience_replay[learning_index])['observation0']
                observation_exp_rep[1]  = (experience_replay[learning_index])['observation1']
                observation_exp_rep[2]  = (experience_replay[learning_index])['observation2']
                observation_exp_rep[3]  = (experience_replay[learning_index])['observation3']
                action_exp_rep          = (experience_replay[learning_index])['action']


                # compute loss:
                # synchronously compute Q_exp_rep; note: Y is asynchronously calculated
                Q_exp_rep = Q_numeric_val(observation_exp_rep, action_exp_rep, DQN)

                # compute loss between real Q (Y) and estimated Q (Q)
                loss = loss_fn(Y_exp_rep, Q_exp_rep)

                sum_loss = sum_loss + loss

                # optimize/backprop with loss:
                # use optimizer with stochastic gradient descent
                optimizer = optim.SGD(DQN.parameters(), lr=LEARNING_RATE)

                # freeze gradients
                optimizer.zero_grad()

                # pytorch backward pass
                loss.backward(retain_graph=True)

                # take optimizer step, propagating error and shifting weights w/ stochastic gradient descent
                optimizer.step()


                # some telemetry
                print(observation)
                print(next_observation)

                print("Loss at timestep {}:     {}".format(time_step, loss))
                print("Y = reward + Q(S',A'):   {}".format(Y_exp_rep))
                print("Q = Q(S,A)):             {}".format(Q_exp_rep))
                print("===" * 35)


            # break at max time_steps
            if time_step == MAX_TIMESTEP:
                done = True

            # update time_step
            time_step += 1


        # total reward  per episode
        reward_accumulator[episode] = sum_reward

        # average loss per  episode
        loss_accumulator[episode] = int(sum_loss/time_step)

    x = EPISODES
    y1 = np.transpose(reward_accumulator)
    y2 = np.transpose(loss_accumulator)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('episodes')
    plt.ylabel('reward/loss')
    plt.legend(['reward', 'loss'])
    plt.show()

def run_testing_operation():
    for timestep in range(MAX_TIMESTEP):
        observation = env.reset()
        action = exploit_action(observation, DQN, env)
        env.render()
        observation, _ = env.step(action)



# TEST SPACE

run_training_operation()

#run_testing_operation()


# todo: eliminate env from exploit action so that implementation is cleaner (maybe faster cuz no calls to other methods)
# todo: make experience replay a class Instead of 2 arrays
# todo: implement multiple episodes as input to the learning algorithm, s.t. we can capture movement of the pole
# QUESTIONS FOR MAX:

# 10 memories isnt long enough
# make it so that we can go under 10 timesteps and learn still (right now it should only learn from greater than 10 timesteps, which may never learn)
# need to be able to learn from more than just the most recent 10 episodes, and also we should learn from previous experiences (not just recent ones()
# learn from sevreal episodes, not just recent one
# openai baselines
