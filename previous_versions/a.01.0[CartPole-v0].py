# learning how to work out at the gym: a.01.0[CartPole-v0]

# change:
# implement random action and algorithmic action I design
# make experiment have modular action
import gym

# env needs to be visible in policy functions
env = gym.make('CartPole-v0')


# return policy given observation object; policy is designed here
def alg_policy(observation):
    # policy: if pole leaning left, move left (put center of gravity below tip)
    #         if pole leaning right, move right (angle positive (observation 2), move right (action 1))
    if observation[2] >= 0: # >= to catch angle = 0 case
        return 1
    else: # (observation[2] < 0
        return 0


# sample random (rnd) policy from space
def rnd_policy(observation):
    return env.action_space.sample()


def experiment(action_policy):
    observation = env.reset()
    done = False
    action = env.action_space.sample()  # first action is random, consecutive should be better

    while done is False:
        env.render(mode='human')           # TODO: SHOULD THIS COME AFTER ACTION IS SELECTED?
        observation, reward, done, info = env.step(action)
        action = action_policy(observation)


# perform experiment:
initial_observation = (0, 0, 0, 0)
experiment(alg_policy(initial_observation))
experiment(rnd_policy(initial_observation))



# some notes about observation/observation_space:
# form: (cart position, cart velocity, pole angle, pole velocity at tip)
#
#                   --- from documentation: ---
#         Type: Box(4)
#         Num	Observation                 Min         Max
#         0	Cart Position             -4.8            4.8
#         1	Cart Velocity             -Inf            Inf
#         2	Pole Angle                 -24°           24°
#         3	Pole Velocity At Tip      -Inf            Inf
#
# some notes about action/action_space:
# form: (push cart to the left, push cart to the right)
#
#                   --- from documentation: ---
#         Type: Discrete(2)
#         Num	Action
#         0	Push cart to the left
#         1	Push cart to the right

