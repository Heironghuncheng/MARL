from pprint import pprint
from envs.micro_grid import MicroGrid
import numpy as np


env = MicroGrid()
env.define_observation_space('./envs/load.csv', './envs/prize.csv', './envs/pv.csv')

env.reset()
for i in range(env.rows):
    if i != 0:
        env.turn()
    for t in range(env.columns):
        action = np.random.uniform(env.action_space[0], env.action_space[1], 1)[0]
        observation, reward, done, info, _ = env.step(action)
        pprint(observation)
        pprint(reward)
        print('\n')
        # if done:
        #     pprint("Episode finished after {} timesteps".format(t + 1))
        #     break

