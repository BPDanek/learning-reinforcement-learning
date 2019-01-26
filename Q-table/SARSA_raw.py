# SARSA on-policy temporal difference
#
# a       b
# c   d   e
# f   g   h

import numpy as np

NUMBER_EPISODES = 1000     # needs update for actual training
LR = 1                # learning rate, needs tuning
DISCOUNT = 1            # discount factor in MDP, needs tuning

STARTING_STATE = 'a'
ENDING_STATE = 'h'

GOAL_ACHIEVED_REWARD = 100   # reward attributed when the ENDING_STATE is achieved
# todo: This ay not be good because it means the network will follow paths it's taken before even if they're not optimal
BASIC_REWARD = 1            # arbitrary reward given for taking a step.



# available states in gridworld
STATES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

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

# define Q table
# since gridworld has no transition probability nor is it continuous
# defining Q(s,a) is easier if done as Q(s,s'). This only works since
# performing an action has a certain outcome, the transition to the next
# state.
#
# the 2d array "Q_table[row][column] represents the weights where
# rows are the current state and columns are the next state

# len(states) returns 7, (a-g). Since we need to describe each state
# to each state, we make a n*n matrix (n = len(states)).
#
# we need to account for the rules of the game. Each node cannot be
# connected to each node
#
# we should do this in another method, and let the weight matrix update
# on its own
def make_table(states):
    # part of algorithm is to initialize the table arbitrarily
    Q_table = np.zeros((len(states), len(states)))
    return Q_table

# given episode, state this function
# computes random number
# derives epsilon from episode number
# uses control flow to call functions that correspond to greedy actions or curous actions
# note: since this is a grid world, actions are actually the name of the next state
#       this means that instead of "take path toward state 2 from state 1" we just say "go to state 2 from state 1"
def select_action_greedy(episode, current_state, Q_table):

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
        return optimal_action(current_state, Q_table)
    else:
        return random_action(current_state, Q_table)

# slects a random action from possible actions at the state
def random_action(current_state, Q_table):

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

# surf through table, for highest option given current state, elect current highest choice,
# then check if that choice is valid. if not valid, re-do search
# this search will become more efficient with time
def optimal_action(current_state, Q_table):

    # the Q_table is mapped as state (x-axis) vs. next_state (y-axis)
    # to compute the optimal action given the current state, we need to go to the entry for the current_state in the
    # table, and then index through every next_state Q_value only returning the highest valued one.

    # go to row, start search. At each element check if is valid, if not, skip, else, compare

    current_state_asint = char_to_index(current_state)

    # need to keep track of index we are in so that we can extract the string of our next state
    # (current state isn't changed here, so theres no need to keep track of that)
    counter = 0

    # the temporary maximum used to identify max weight in Q_table
    compare_val = 0

    # when a comparison, and a reassignment is done, the counter will be stored here to denote the current
    # biggest number. Will only be overwritten if a new max is found
    # if this number is never reassigned, it should remain zero, indicating no best choice is clear and that
    # we need to act randomly
    compare_address = -1

    # this loop should go through the possible actions, and rbing out the index of the largest one
    # since row denotes current state, we loop through all possible complementary next states
    for element in Q_table[current_state_asint]:

        # if the next state (action) is adjacent, we will proceed to check the Q val
        # if it is not, we'll just go ahead and skip it
        if is_valid_choice(current_state, index_to_char(counter)):

            if compare_val > element:
                compare_val = element
                compare_address = counter

        counter += 1

    if compare_address != -1:
        # compare_address should hold the highest valued action given the current state
        optimal = index_to_char(compare_address)
        return optimal
    else:
        random = random_action(current_state, Q_table)
        return random

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




# main:

# test = np.matrix('1,2,3; 4,5,6; 7,8,9')
# print(test[1])
#
#
# for element in test[0]:
#
#     if is_valid_choice('c', index_to_char(counter)):
#
#         if compare_val > element:
#             compare_val = element
#             compare_address = counter
#
#     counter += 1



# core algorithm

# initialize Q_table arbitrarily* (np.zeros)
Q_table = make_table(STATES)

cumilative_reward = 0  # todo: is this in the right place?

for episode in range(NUMBER_EPISODES):

    print("episode: {} \n{}".format(episode, Q_table))

    # initialize state in each episode as current start
    state = STARTING_STATE

    # select action_0 based on epsilon greedy
    action = select_action_greedy(episode, state, Q_table)

    # this flag will become true when reward for achieving goal is achieved, it's the end condition for this loop
    episode_terminate = False

    # in this loop we'll enter have time steps
    time_step = 0
    while episode_terminate is False:

        # print("time = {}".format(time_step))
        time_step += 1


        # take action computed at start of episode, observe outcome (reward, next state)
        next_state, reward = step(state, action)

        # update end condition (which is dependant on reward)
        # if the goal has been achieved, terminate the episode
        # achieving the goal returns a unique reward
        if reward == GOAL_ACHIEVED_REWARD:
            episode_terminate = True

        # accumulate reward
        cumilative_reward += reward

        # choose next action from the new state using epsilon greedy
        next_action = select_action_greedy(episode, next_state, Q_table)

        # update Q value; this is done using the Q_table array, which uses indices to read/write values. Since
        # actions and states are described using strings, we need to convert it to the form that the
        # Q_table can use (integers/indices)
        state_asint = char_to_index(state)
        action_asint = char_to_index(action)
        next_state_asint = char_to_index(next_state)
        next_action_asint = char_to_index(next_action)

        # computation portion
        Q_table[state_asint][action_asint] = Q_table[state_asint][action_asint] + (LR) * (cumilative_reward + \
            (DISCOUNT * Q_table[next_state_asint][next_action_asint]) -  Q_table[state_asint][action_asint])

        if (episode == (NUMBER_EPISODES - 1)) or (episode % 100):
            print("\t{} --> {}".format(state, action))

        # update state and action
        state = next_state
        action = next_action

# initialize Q(s,a) arbitrarily
# repeat for each episode:
#   initialize state
#   chose action_0 from state_0 using policy using epsilon
#   while in episode (for each time step in episode):
#       take action_0, observe reward and next state_1
#       choose next action_1 form next state_1 using epsilon
#       update Q: Q(state_0, action_0) = Q(state_0, action_0) + (LR)*(reward + discount*Q(state_1, action_1) - Q(state_0, action_0))
#   until S is terminal

