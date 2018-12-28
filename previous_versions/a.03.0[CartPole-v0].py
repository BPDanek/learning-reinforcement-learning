# learning how to work out at the gym: a.03.0[CartPole-v0]
# Q-Table attempt 1

# organization: (from previous)
# produce environment (env) of CartPole
# produce an experiment using globals and the passed param: Policy Signal
#
# in each experiment we run the number of episodes, each episode with the policy signal known
#       in the episode, a multiplexer is called with the signal as parameter, which returns an action
#           the purpose of the signal is to determine what action to take (decision policy)

import gym  # CartPole-v0 dependancy

# some constants
NUM_EPISODES = 2
MAX_REWARD = 100

RANDOM_SELECT =         0
ALGORITHMIC_SELECT =    1
QTABLE_SELECT =         2

# use single global env that gets reset per episode
env = gym.make('CartPole-v0')


# BASELINE: sample random (rnd) policy from space
def rnd_policy(observation):
    return env.action_space.sample()


# BASELINE: return policy given observation object; policy is designed here
def alg_policy(observation):
    # policy: if pole leaning left, move left (put center of gravity below tip)
    #         if pole leaning right, move right (angle positive (observation 2), move right (action 1))
    if observation[2] >= 0: # >= to catch angle = 0 case
        return 1
    else: # (observation[2] < 0
        return 0


# Q-TABLE: implement a means of producing some learned policy in the program
#   each possible state, action pair will have a determined (quality) Q-value
#   with a complete table of Q values, we can safely be put into any state and know what the best next move is
#   with that being said, the difficulty is learning how to determine these Q-values
#   TODO: How are Q values learned?
# def qtable_policy(observation):
    # TODO: implementation


# returns action decided by signal, 0 for random, 1 for algorithmic
def action_mux(policy_signal, observation):

    if policy_signal == ALGORITHMIC_SELECT:
        return alg_policy(observation)

    elif policy_signal == QTABLE_SELECT:
        return qtable_policy(observation)

    else: # signal = 0 (RANDOM_SELECT)
        return rnd_policy(observation)


def experiment(policy_signal):

    cumilative_reward_across_episodes = 0

    for episode in range(NUM_EPISODES):

        done = False
        total_reward = 0  # initially reward is zero in each experiment

        observation = env.reset()

        # run episode simulation, collect reward
        while done is False:

            env.render()

            action = action_mux(policy_signal, observation)

            observation, t_reward, done, info = env.step(action)

            total_reward = total_reward + t_reward


            # debugging info:
            # print('a_:', action)
            # print('o2:', observation[2])  # object, 4 values shown below
            # print('d_:', done)
            # print('tr:', total_reward)
            # print('-------------------------------------------------------|')

            if total_reward > MAX_REWARD:
                print('Max reward occured, '
                      'consider episode number {} of {} to be complete.'.format(episode + 1, n_episodes))
                break

        print('reward: {}'.format(total_reward))
        cumilative_reward_across_episodes += total_reward

    print('average reward: {}'.format(cumilative_reward_across_episodes/NUM_EPISODES))
    print('-'*25)




print('random policy: ')
experiment(RANDOM_SELECT)
print('algorithmic policy:')
experiment(ALGORITHMIC_SELECT)
# print('algorithmic policy:')
# [alg_policy]


# saves pyglet some worry
env.close()