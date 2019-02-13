# this is the Q-network* version of the toy example (SARSA_raw)
# SARSA on-policy temporal difference


import numpy as np
np.set_printoptions(precision=1, suppress=True)
import random
import torch.nn as nn
import torch
import torch.nn.functional as F

NUMBER_EPISODES = 20000     # needs update for actual training
LR = 0.1                # learning rate, needs tuning
DISCOUNT = 1            # discount factor in MDP, needs tuning

STARTING_STATE = 'a'
ENDING_STATE = 'h'

GOAL_ACHIEVED_REWARD = 100   # reward attributed when the ENDING_STATE is achieved
# todo: This ay not be good because it means the network will follow paths it's taken before even if they're not optimal
BASIC_REWARD = -1            # arbitrary reward given for taking a step.

# available states in gridworld
STATES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class deep_q_network(nn.Module):

    def __init__(self):
        super(deep_q_network, self).__init__()  # inherit properties and behaviour from neural network module pytorch

        # input data is: state
        # output data is: next_state
        self.fc1 = nn.linear(in_features=1, out_features=2)
        self.fc2 = nn.linear(in_features=3, out_features=4)
        self.fc3 = nn.linear(in_features=3, out_features=2)
        self.fc4 = nn.linear(in_features=3, out_features=1)

    def forward(self, state):
        state = self.fc1(state)
        state = self.fc2(state)
        state = self.fc3(state)
        state = self.fc4(state)
        return state


# this structure is tied heavily to the layout of the problem. The dict allows description of the rules of the game.
# example: definition 'a' has 'c' adjacent to it, and thus legal to transition to. 'b' has 'e', 'e' has 'b','d','h'
# this funciton is used to check legality of actions selected later on
# note: to change the game, this dict needs to be augmented to follow the other game
def house_graph():
    graph = {
        'a' : ['c'],
        'b' : ['e'],
        'c' : ['a', 'd', 'f'],
        'd' : ['c', 'e', 'g'],
        'e' : ['b', 'd', 'h'],
        'f' : ['c', 'g'],
        'g' : ['d', 'f', 'h'],
        'h' : ['e', 'g']
    }
    return graph

# given episode, state this function
# computes random number
# derives epsilon from episode number
# uses control flow to call functions that correspond to greedy actions or curous actions
# note: since this is a grid world, actions are actually the name of the next state
#       this means that instead of "take path toward state 2 from state 1" we just say "go to state 2 from state 1"
def select_action_greedy(episode, current_state, Q_network):

    normal_random = np.random.rand()
    epsilon = 1 / (episode + 1)  # per sutton/barto's example

    # since we want increased random actions in the beggining, and lesser with time, and our epsilon will decrease
    # with episodes, our expresison should be:
    # "When random number is greater than the decreasing epsilon, exploit"
    # todo: see epiphany below:
    #  an issue I noticed: wouldn't this mean that if our epsilon declined too quickly,
    #  our exploration would be done very thoroughly in the beggining but less so toward the end?
    #  we should have substantially more exploration than there are finite states so that we thoroughly explore all
    #  possible actions.
    #  epiphany continued: this won't actually be an issue since exploration probability will decreaser per *episode*
    #  this means that for an entire episode itll explore, therefore adequatley showing off the entire space. The first
    #  episode will just take a long time/have lots of steps


    if normal_random > epsilon:
        return optimal_action_QN(current_state, Q_network)
    else:
        return random_action(current_state, Q_network)

# slects a random action from possible actions at the state
def random_action(current_state, Q_network):

    # call a dict showing all possible actions
    # the 2d array full of weights is good for writing and calling weights, but not good for showing which transitions
    # are legal by the game rules. We use the dict to check that. Since random select does not check for weight values
    # we can make our random sample purely from the dictionary (which, again, identifies purely which choices are legal)
    graph = house_graph()
    # define a dict to sample randomly (like prev.) sarsa implementation w/ dict.
    # return string current action


    # determine length of possible actions, subtract 1 to start at 0
    possible_actions = len(graph[current_state])

    # determine a random number within the specified range, including possible actions
    random_index = np.random.randint(0, possible_actions)

    # select the value at the random index
    selection = graph[current_state][random_index]

    # note the return val of this is a string! you need to keep this in mind for future cases
    return selection

# previously:
# surf through table, for highest option given current state, elect current highest choice,
# then check if that choice is valid. if not valid, re-do search
# this search will become more efficient with time
#
# currently:
# use network to generate optimal output based on current training data
# output needs to be returned in similar format as random_action (character)
def optimal_action_QN(current_state, Q_network):
    Q_network.zero_grad()
    return Q_network.forward(current_state)


# todo: assess whether this belongs in DQN
# given a state (str), and a candidate (str) this method evaluates whether the candidate is a legal
# next step to take in the game based on the rules. This function copies the house graph
# and looks at all adjacent nodes to the one you're in by searching the dict. for the state
# and returning all of the keys. It then loops through the keys to see if the candidate
# is in that set of keys, if it is, then the candidate could be considered legal given the state
def is_valid_choice(state, candidate):
    # working copy of game
    graph = house_graph()

    # default to false, will only change if there is a match
    flag = False

    # keys to dictionary entry: state
    options = graph[state]

    # loop through every single entry and check if one of the entries matches candidate
    for option in options:
        if option == candidate:
            # return val augmented in case of match
            flag = True

    return flag

# given state, action (next state)
# return reward, next state (the action in this case)
# next_state, reward = step(state, action)
def step(state, action):

    # compute reward, give basic for timestep, give better for achieving goal
    if action == ENDING_STATE:
        reward = GOAL_ACHIEVED_REWARD
    else:
        reward = BASIC_REWARD

    # next state is the action passed in. This is explained in the design of the problem in the beggining, but
    # since we have finite states that are adjacent to each other, describing the action taken is the same as
    # describing the next state from the current one
    next_state = action

    return next_state, reward

# given character or string
# return integer that describes this
# this is a helper function required by my data structure choice for the Q table
# Q_values are stored as integers within a 2D array, and thus need to be accessed with their indeces. The issue
# is that I have described the rows/columns of the array syntactically
def char_to_index(state_asstr):
    # returns ascii value of state # which is actually a string w/ the character denoting the state ord('a') - 97 = 0

    state_asint = ord(state_asstr) - 97
    return state_asint

# the opposite of @char_to_index above, python's ascii decoder makes this mostly easy, but I've
# taken the extra step to make this a function for the sake of readability
def index_to_char(state_asint):

    # returns index of array as alphabetical equivalent where a = 0, b = 1, c = 2 ...
    state_asstr = chr(state_asint + 97)
    return state_asstr


# for stronger training, the network start location should randomize
def random_starting_state(ending_state_asint):

    start_state = ending_state_asint

    # loop should enter because start_state initialized as ENDING_STATE, (condition for loop cont.) and break the second
    # it is no longer the same, which is exactly what we need
    while start_state is ending_state_asint:
        start_state = random.randint(0, len(STATES) - 1) # randint returns start <= rand <= end (inclusive)

    start_state_aschar = index_to_char(start_state)

    return start_state_aschar


# returns the Q value of the state, action pair, as if a Q table would
# this is a necessary component for quantifying loss of the neural network
def value_funct_Q(state, action, network):
#TODO: The issue encountered here is that I need to make a structure for estimating the Q VALUE not the outcome of the Q value. I encountered
# this same exact problem last time. wtf

# main:


# core algorithm

# initialize Q_table arbitrarily (np.zeros)
Q_network_current = deep_q_network()


# make optimizer object that will be used throughout learning
optimizer = torch.optim.SGD(Q_network_current.parameters(), lr=LR)

for episode in range(NUMBER_EPISODES):

    # initialize state in each episode as current start
    state = random_starting_state(char_to_index(ENDING_STATE))  # new starting position each episode

    print("episode: {} --- {}\n{}".format(episode,state, Q_network_current))

    # this flag will become true when reward for achieving goal is achieved, it's the end condition for this loop
    episode_terminate = False

    # in this loop we'll enter have time steps
    time_step = 0
    while episode_terminate is False:

        time_step += 1

        # some verbosity associated w/ number of episodes
        if (episode == (NUMBER_EPISODES - 1)) or (episode % 100):
            print(" {}-->{} ".format(state, action))

        # terminate episode if stuck in loop
        if time_step == 50:
            episode_terminate = True

        # select action given current state
        action = select_action_greedy(episode, state, Q_network_current)

        # take action, observe next state and reward
        next_state, reward = step(state, action)

        reward_from_current_Q = reward  # reward from current state, action pair:

        # update end condition (which is dependant on reward)
        # if the goal has been achieved, terminate the episode
        # this check works since achieving the goal returns a unique reward
        if reward == GOAL_ACHIEVED_REWARD:
            episode_terminate = True


        # compute loss
        # update weight

        # components of the loss calculation are derived from the bellman equation
        # the bellman equation describes optimization problems in a recursive manner
        # the best possible action is one that maximizes all future reward; Q is defined as
        # reward + reward of next state (which entails the reward of the next next state and so on)
        current_Q = value_funct_Q()(state, action, Q_network_current)

        next_next_state = optimal_action_QN(next_state, Q_network_current)  # note that optimal_action returns the next good state, not an actual action

        theoretical_Q = reward_from_current_Q + (DISCOUNT*value_funct_Q(next_state, next_next_state, Q_network_current))


        # weight udpate:
        optimizer.zero_grad()   # prevent changes to gradient from backward pass

        # loss function is mean square error
        # means square error between output and target (mse(output - target)) = (mse(q(s,a,current_network) - q(s,a, previous network)
        loss = nn.MSELoss(current_Q, theoretical_Q)
        loss.backward()
        optimizer.step()



        # SARSA
        # computation portion (loss)
        # from text:
        # Q_table[state_asint][action_asint] = Q_table[state_asint][action_asint] + (LR) * (reward + (DISCOUNT * Q_table[next_state_asint][next_action_asint]) -  Q_table[state_asint][action_asint])
        #
        # from lecture: (this one has success)
        # Q_table[state_asint][action_asint] = (1 - LR)*(Q_table[state_asint][action_asint]) + (LR)*(reward + DISCOUNT*Q_table[next_state_asint][next_action_asint])


        # final step in formal Q-learning algorithm is incrementing the state
        state = next_state
