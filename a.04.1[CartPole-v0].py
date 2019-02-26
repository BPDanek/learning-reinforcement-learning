# learning how to work out at the gym: a.04.1[CartPole-v0]
# todo: Implement experience replay into this bitch

# IMPORTS ---------------------------------------------------------------------------------------------------------

import numpy as np
import gym
import torch.nn as nn
from torch import optim as optim
import torch

import matplotlib.pyplot as plt

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------
LEARNING_RATE = 0.01

# number of episodes over which to perform learning of Q estimator
EPISODES = 10
MAX_TIMESTEP = 50

# every __ episodes compute loss, and propogate
EXP_REPLAY_NUM = 10


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

# Algorithm Implementation ----------------------------------------------------------------------------------------


env = gym.make('CartPole-v0')  # CartPole simulation environment from gym
# reset() generates a single observation, which is (4,) with each element a uniform value between [-0.05, 0.05]

# make a DQN object
DQN = DQNetwork()

# declare MSE loss function once
loss_fn = nn.MSELoss()

# declare exp. replay
experience_replay_sa = np.zeros((EXP_REPLAY_NUM, 5)) # exp_replay_num*3(columns are Y, state, action)
experience_replay_Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # iteration of exp_replay_num (assuming a memory ios 10 elements


# used for debugging, shows loss change throughout episodes
loss_accumulator = (np.zeros((EPISODES, (MAX_TIMESTEP))))

def run_training_operation():

    for episode in range(EPISODES):

        print("\nEpisode: ", episode)

        time_step = 0               # time-step of current episode
        buffer_counter = 0          # counter to record buffer index for experience replay
        done = False                # flag that breaks out of episode; changed when env returns "done"
        loss_accumulator_increment = 0


        observation = env.reset()   # resetting environment returns initial observation

        # primary algorithm iteration within episode; reset each episode
        # select action, learn something
        # action determined by exploration-exploitation trade-off
        # learning done by NN instance, sample to learn comes from direct memory
        while not done:


            # exploration-exploitation tradeoff epsilon used to decide when to exploit and when to explore
            epsilon_rand = np.random.rand()

            # exploration decline is defined as such; seen in sutton/barto's SARSA
            current_explore_limit = 1/(episode + 1)

            # explore limit decreases with time. epsilon greater than limit, explore; else explopit
            if epsilon_rand <= current_explore_limit:
                action = exploit_action(observation, env, DQN)
            else:
                action = env.action_space.sample()  # take random (exploratory) action


            # take step, observe reward, check for done
            # this step updates: state = state++
            next_observation, reward, done, _ = env.step(action=action)


            # TODO: remove this once building the code is done
            # break at a time step limit
            if time_step == MAX_TIMESTEP:
                done = True


            # learning step
            # compute corresponding optimal action "next_action" to its state "next_state"
            # these values will be used to calculate target value of NN
            next_action = exploit_action(next_observation, env, DQN)

            # target Q_val = reward + Q(S',A), will be stored in buffer
            Y = reward + Q_numeric_val(next_observation, next_action, DQN)

            # store state, action, Y in buffer, compute loss every few episodes
            state_action_concat = torch.FloatTensor(np.append(observation, action))


            # STORE INTO BUFFER actual_Q, observation, action for current timestep
            # these values to be computed in future
            experience_replay_sa[buffer_counter] = state_action_concat
            experience_replay_Y[buffer_counter] = Y


            # increment buffer counter before it's reset within "if" so that it can be zero'd easier
            buffer_counter += 1

            # loss @ every EXP_REPLAY_NUM timesteps:
            if (buffer_counter % EXP_REPLAY_NUM == 0) and buffer_counter is not 0:

                print("\nbuffer counter is: {}\n".format(buffer_counter))

                # reset so that we can fill it again
                buffer_counter = 0


                experience = 0

                while experience < EXP_REPLAY_NUM:

                    # actual Q value
                    print(experience)
                    Y = experience_replay_Y[experience]
                    state = experience_replay_sa[experience][:4]
                    action = experience_replay_sa[experience][4]

                    # neural network estimated Q
                    Q = Q_numeric_val(state, action, DQN)

                    # loss between real Q (Y) and estimated Q (Q)
                    loss = loss_fn(Y, Q)

                    # need to have this funky addition mechanism because we propagate loss asynchronously, but
                    # want to populate the loss accumulator in a synchronous order
                    loss_accumulator[episode][experience + loss_accumulator_increment] = loss

                    # use optimizer with stochastic gradient descent
                    optimizer = optim.SGD(DQN.parameters(), lr=LEARNING_RATE)

                    # freeze gradients
                    optimizer.zero_grad()

                    # pytorch backward pass
                    loss.backward()

                    # take optimizer step, propagating error and shifting weights w/ stochastic gradient descent
                    optimizer.step()

                    print(observation)
                    print(next_observation)

                    print("Loss at timestep {}:     {}".format(time_step, loss))
                    print("Y = reward + Q(S',A'):   {}".format(Y))
                    print("Q = Q(S,A)):             {}".format(Q))
                    print("===" * 35)

                    # update experience counter
                    experience += 1

                # increment this value that allows us to add things to the loss_accumulator better
                loss_accumulator_increment += 1


            # update time_step
            time_step += 1

    # print information from debugging
    x = range(MAX_TIMESTEP)
    y_0 = loss_accumulator[7]
    y_1 = loss_accumulator[8]
    y_2 = loss_accumulator[9]

    plt.plot(x, y_0)
    plt.plot(x, y_1)
    plt.plot(x, y_2)

    plt.xlabel('timesteps')
    plt.ylabel('loss_accumulator object')
    plt.legend(['episode n', 'episode n+1', 'episode n+2'])

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