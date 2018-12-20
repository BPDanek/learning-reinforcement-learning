# learning how to work out at the gym: a.00.1[CartPole-v0]
import gym

env = gym.make('CartPole-v0')

# observation = env.reset() # returns initial observation
# print(observation)

for iter in range(1):

    observation = env.reset()
    done = False

    while done is False:
        env.render(mode='human') # renders a 0.2 s (or ms i can't remember right now) moment of the simulation

        action = env.action_space.sample()
        print('a:', action) # object, 2 values shown below

        observation, reward, done, info = env.step(action)

        print('o:', observation) # object, 4 values shown below
        print('r:', reward) # float, in CartPole-v0, reward is 1 for every time step, including termination step
        print('d:', done) # boolean
        print('-------------------------------------------------------|')
        iter = iter + 1
        if done:
                print("completed after {} timesteps".format(iter))
                print('-------------------------------------------------------|')

# some notes about observation/observation_space:
# form: (cart position, cart velocity, pole angle, polve velocity at tip)
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

