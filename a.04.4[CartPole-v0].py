# learning how to work out at the gym: a.04.4[CartPole-v0]

# IMPORTS ---------------------------------------------------------------------------------------------------------
import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
import torch
from tensorboardX import SummaryWriter


SAVE_PATH = "/Users/denbanek/PycharmProjects/dqn_project_dir/learning-reinforcement-learning/dqn_weights/100.pth"
JSON_LOGDIR = "./logdirectory/learning_scalars.json"
LOGDIR = "./logdirectory"

# HYPERPARAMETERS ------------------------------------------------------------------------------------------------
LEARNING_RATE = 0.005

# note: the discount factor isn't actually used in this problem (yet)
DISCOUNT = 0.99

# number of episodes over which to perform learning of Q estimator
EPISODES = 10000
MAX_TIMESTEP = 100

# we want the number of memories to remember to be relatively small, as only the most recent memories will be
# remembered due to the Queue-like object that stores them (Buffer)
NUM_EPISODES_TO_REMEMBER = 5
EXPERIENCE_REPLAY_SIZE = 6

# learning is performed in batches where loss is accumulated throughout batch, this accumulated loss is the
# loss that is propagated
BATCH_SIZE = 50

# every 5 pisodes we go through learning process
LEARNING_INTERVAL = 5

# make structure for visualizing data:
writer = SummaryWriter(LOGDIR)

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

        self.fully_connected1 = nn.Linear(in_features=5, out_features=16)
        self.fully_connected2 = nn.Linear(in_features=16, out_features=1)
        # self.fully_connected3 = nn.Linear(in_features=3, out_features=2)
        # self.fully_connected4 = nn.Linear(in_features=2, out_features=1)



    # todo: identify if these activations are reasonable
    def forward(self, input_data):
        out = F.relu(self.fully_connected1(input_data))
        out = (self.fully_connected2(out))
        # out = (self.fully_connected3(out))
        # out = self.fully_connected4(out)
        #writer.add_histogram("fwd pass", out)
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

    DQN.zero_grad()

    Q_value_L = DQN(combined_in_L)
    Q_value_R = DQN(combined_in_R)

    if Q_value_L > Q_value_R:
        return 0

    elif Q_value_L < Q_value_R:
        return 1

    else:
        return env.action_space.sample()

"""
This method returns the numeric value of the optimal Q value given a state, network object that is trained.
Since this method may have no optimal Q value, we need the env so that we can sample the action space randomly
"""
def Q_numeric_val(observation, action, DQN, return_graph=False):
    # since DQN intakes state and action concatenation, we can recreate the correct input to the network, assuming
    # the state of the network is unchanged

    # input is the state_action pair concatenate
    input = torch.FloatTensor(np.append(observation,  action))

    # zero gradient buffers, avoid autograd differentiation via current input. We don't want to overwrite buffers
    # since we need their information (gradients) still
    DQN.zero_grad()

    temp = DQN(input)

    return temp


def push_and_pop(Y, obs, act, exp_replay, index):
    # need to translate s.t. we could insert into index easily
    memory = np.zeros(EXPERIENCE_REPLAY_SIZE)

    # use Y as scalar
    Y = Y.detach().numpy()

    memory[0] = float(Y)
    for element in range(obs.size):
        memory[Y.size + element] = obs[element]
    memory[Y.size + len(obs)] = float(act)

    # fill replay with memories; when  itll be full we'll flush it
    exp_replay[index] = memory

    index += 1

    return index, exp_replay


# Algorithm Implementation ---------------------------------------------------------------------------------------

# make a DQN object:
env = gym.make('CartPole-v0')
DQN = DQNetwork()


def run_training_operation():
    # means of representing learning success
    loss_accumulator = np.ndarray((1))

    # declare optimizer
    optimizer = optim.SGD(DQN.parameters(), lr=LEARNING_RATE)

    # declare MSE loss function:
    loss_fn = nn.MSELoss()

    # declare experience replay:

    # experience replay should store a recent set of memories
    # the data stored should resemble Y (asynchronous), state, action (which derive Y). The idea is that
    # the Q value will be derived from the state and action synchronously, and Y will not. Training should
    # be done in batches, where loss is accumulated throughout the batch and then propogated.

    # formal structure, the first argument represents a the number of recent memories to remember, it is computed
    # in the "constants" section of the program.
    # the second argument represents the information that should be stored, which was described earlier.
    experience_replay = np.zeros(((NUM_EPISODES_TO_REMEMBER * MAX_TIMESTEP), EXPERIENCE_REPLAY_SIZE))

    # index that will show what element one can insert memory to
    pnp_idx = 0

    # episode training loop:
    for episode in range(EPISODES):

        print("\nEpisode: ", episode)

        Y_acc = 0
        reward_acc = 0
        loss_acc = 0
        Q_acc = 0

        # declarations/initializations for episodes
        # time-step of current episode
        time_step = 0

        # flag that breaks out of episode; changed when env returns "done"
        done = False

        # resetting environment returns initial observation
        observation = env.reset()

        # episode run:
        while (done is not True) and (time_step < MAX_TIMESTEP):


            # select action to take:
            # generate random number to be epsilon rand
            epsilon_rand = np.random.rand()

            # set limit for exploration: if above limit, exploit, if below, explore
            # limit declines with time, so being above it becomes more likely
            # based loosely off of Nature paper of DQN in atari  games, we set a minimum explore limit of 0.1 after a
            # certain point, here  its after  1/20th of  the episodes is  complete, in the paper its after 10 episodes
            # which to me seemed unscalable, and  short, so i doubled the  length  and scaled it based on  episode
            # hyperparameter.

            if episode < int(0.3*EPISODES):
                current_explore_limit = 0.99
            elif episode < int(0.6*EPISODES):
                current_explore_limit = 0.5
            elif episode < int(0.8*EPISODES):
                current_explore_limit = 0.3
            else:
                current_explore_limit = 0.1

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
            reward_acc += reward

            # requisites for bellman equation:
            # compute corresponding optimal action "next_action" to its state "next_state"
            # these values will be used to calculate target value of NN
            next_action = exploit_action(next_observation, env, DQN)

            # filling exp replay:
            # target Q_val = reward + Q(S',A), will be stored in buffer
            # note: in other RL environments we would have a Y calculation for non-terminal and terminal reward
            #       since there is no Q_numeric if we are done
            Y = reward + (DISCOUNT * Q_numeric_val(next_observation, next_action, DQN))
            Y_acc += Y

            # fill the data structure with info for Y, observation, action
            # Y = reward + Q(s', a') <-- value stored
            # Q = Q(s, a) <-- these state & action are the ones stored

            # push memory onto buffer
            # note, pnp_idx denotes the number of memories we have in buffer. most of the time, buffer is full, and
            # since it behaves like a queue, it will
            pnp_idx, experience_replay = push_and_pop(Y, observation, action, experience_replay, pnp_idx)

            # update observatoin
            observation = next_observation

            # update time_step
            time_step += 1
            # END WHILE

        # learning step
        # learning step performed every few episodes, ideally whne memory is full
        if (episode % NUM_EPISODES_TO_REMEMBER) is 0 and (episode != 0):

            # policy loss object will be the sum of losses within batch
            policy_loss = []

            for batch_num in range(pnp_idx):

                # sample random episode from memory buffer
                learning_index = np.random.randint(0, pnp_idx)

                # read memory data from data structure
                Y_exp_rep = experience_replay[batch_num][0]
                observation_exp_rep = experience_replay[batch_num][1:5]
                action_exp_rep = experience_replay[batch_num][5]

                # compute loss:
                # synchronously compute Q_exp_rep; note: Y is asynchronously calculated
                Q_exp_rep = Q_numeric_val(observation_exp_rep, action_exp_rep, DQN)
                Q_acc += Q_exp_rep

                # compute loss between real Q (Y) and estimated Q (Q)
                # mem_loss is the loss of the single memory
                # batch_loss is the loss in the entire batch
                mem_loss = loss_fn(Y_exp_rep, Q_exp_rep)
                policy_loss.append(mem_loss)

                np.append(loss_accumulator, mem_loss.detach().numpy())

            # optimize/backprop with cumilative loss:
            # use optimizer with stochastic gradient descent

            # freeze gradients
            optimizer.zero_grad()

            # sum of loss over batch of memories
            sum_loss = sum(policy_loss)
            loss_acc += sum_loss

            # pytorch backward pass
            sum_loss.backward()

            testing_result = run_testing_operation()

            # flush experience replay:"
            pnp_idx = 0


            writer.add_scalar("testing within loss funct", testing_result, global_step=episode)

            # take optimizer step, propagating error and shifting weights w/ stochastic gradient descent
            optimizer.step()
        writer.add_scalar("Y_acc_train", Y_acc, global_step=episode)
        writer.add_scalar("reward_acc_train", reward_acc, global_step=episode)
        writer.add_scalar("loss_acc_train", loss_acc, global_step=episode)


    writer.export_scalars_to_json(JSON_LOGDIR)
    # save to local directory
    torch.save(DQN.state_dict(), SAVE_PATH)

    return loss_accumulator


def run_testing_operation():
    num_runs = 10
    # DQN = DQNetwork()
    # DQN.load_state_dict(torch.load(SAVE_PATH))
    average_time = 0


    print("number of iterations: {}".format(EPISODES))
    for run in range(num_runs):
        observation = env.reset()
        time = 0
        reward_acc_test = 0

        for time_step in range(9000):

            #action = env.action_space.sample()
            action = exploit_action(observation, env, DQN)
            observation, reward, done, info = env.step(action)

            reward_acc_test += reward

            if done is True:
                break
            time += 1
        writer.add_scalar("reward_acc_test", reward_acc_test, global_step=run)

        average_time += time
        print("time for run {}: {}".format(run, time))
    avg = average_time/num_runs
    print("average time: ", (avg))
    return avg

# TEST SPACE

run_training_operation()

run_testing_operation()

# todo: eliminate env from exploit action so that implementation is cleaner (maybe faster cuz no calls to other methods)
# todo: make experience replay a class Instead of 2 arrays
# todo: implement multiple episodes as input to the learning algorithm, s.t. we can capture movement of the pole
# QUESTIONS FOR MAX:

writer.close()