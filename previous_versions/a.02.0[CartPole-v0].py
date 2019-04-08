# learning how to work out at the gym: a.02.0[CartPole-v0]

# make classes
#   random agent
#   algorithmic agent (simple solution)
#   future: ML agent
#

import gym

env_name = 'CartPole-v0'

class RandomAgent:

    def __init__(self, env_name, max_episodes):
        self.env = gym.make(env_name)
        self.max_episodes = max_episodes

    def run(self):

        episode_logbook = [self.max_episodes]

        for episode in range(self.max_episodes):

            done = False
            sum_reward = 0
            iterator = 0

            while done is False:

                iterator += 1

                _ = self.env.reset() # reset, return initial observation

                if (iterator % 10) == 0:
                    self.env.render(mode='human')

                action = self.env.action_space.sample()

                _, reward, done, _ = self.env.step(action)

                sum_reward += reward
                episode_logbook[episode] = sum_reward

            print('episode {} score: {}'.format(episode, episode_logbook[episode]))


#class AlrorithmicAgent:

 #   def policy:

  #  def run:

random_agent = RandomAgent(env_name, 2)

random_agent.run()
