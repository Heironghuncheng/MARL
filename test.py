# coding=utf-8

import tqdm
import gymnasium as gym
import tensorflow as tf
# from envs.func import SecFunc
from envs.micro_grid import MicroGrid
from newa2c_test import Agent
import random

if __name__ == "__main__":
    # turn = 0
    writer = tf.summary.create_file_writer("./log")
    agent = Agent(32, 0, -500, 500, 0, 1000, 0, 1000, 0.0035, 0.0025, 8, 300, 0.05)
    env = MicroGrid(writer)
    env.define_observation_space('./envs/load.csv', './envs/prize.csv', './envs/pv.csv')
    with tqdm.trange(30000) as t:
        # env.render()
        for i in t:
            first_state = env.reset(random.randint(0, 1369))
            first_state = tf.expand_dims(first_state, 0)
            reward = agent.multi_object(first_state, env, writer, i)
            t.set_description(f'Episode {i} reward {reward}')
            with writer.as_default():
                tf.summary.scalar('all_reward', float(reward), step=i)
                tf.summary.flush()
        writer.close()

