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
        action = []
        for j in range(3):
            action.append(np.random.uniform(env.action_space[j][0], env.action_space[j][1], 1)[0])
        action = tuple(action)
        observation, reward, done, info = env.step(action)
        pprint(observation)
        pprint(reward)
        print('\n')
        # if done:
        #     pprint("Episode finished after {} timesteps".format(t + 1))
        #     break

