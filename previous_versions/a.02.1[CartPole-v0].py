# learning how to work out at the gym: a.02.1[CartPole-v0]

# change:
# implement random action and algorithmic action I design
# make experiment have modular action
import gym
import numpy as np

# env needs to be visible in policy functions
env = gym.make('CartPole-v0')

# some constants
NUM_EPISODES = 3
MAX_REWARD = 100
RANDOM_SELECT = 0
ALGORITHMIC_SELECT = 1

# return policy given observation object; policy is designed here
def alg_policy(observation):
    # policy: if pole leaning left, move left (put center of gravity below tip)
    #         if pole leaning right, move right (angle positive (observation 2), move right (action 1))
    if observation[2] >= 0: # >= to catch angle = 0 case
        # print(observation[2], '-- 1')
        return 1
    else: # (observation[2] < 0
        # print(observation[2], '-- 0')
        return 0


# sample random (rnd) policy from space
def rnd_policy(observation):
    return env.action_space.sample()


# returns action decided by signal, 0 for random, 1 for algorithmic
def action_mux(policy_signal, observation):
    if policy_signal == 0:
        return rnd_policy(observation)
    else: # signal = 1
        return alg_policy(observation)


def experiment(policy_signal, n_episodes, max_reward):

    for episode in range(n_episodes):

        done = False
        inf_loop_break = 0
        total_reward = 0  # initially reward is zero in each experiment

        observation = env.reset()

        # run episode simulation, collect reward
        while done is False:

            # bad logic catcher
            inf_loop_break += 1
            if inf_loop_break == 400:
                print('NoExitError: An infinite loop has formed in the experiment')
                break

            env.render()

            # print('io:', observation)

            action = action_mux(policy_signal, observation)

            observation, t_reward, done, info = env.step(action)

            # add the reward of the time step to the reward
            total_reward = total_reward + t_reward

            # print('a_:', action)
            # print('o2:', observation[2])  # object, 4 values shown below
            # print('d_:', done)
            # print('tr:', total_reward)
            # print('-------------------------------------------------------|')

            if total_reward > max_reward:
                print('Max reward occured, '
                      'consider episode number {} of {} to be complete.'.format(episode + 1, n_episodes))
                break

        # after simulation complete, update reward in logbook
        print('reward: {}'.format(total_reward))


print('random policy: ')
experiment(RANDOM_SELECT, NUM_EPISODES, MAX_REWARD)
print('algorithmic policy:')
experiment(ALGORITHMIC_SELECT, NUM_EPISODES, MAX_REWARD)

env.close()