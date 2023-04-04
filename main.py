# coding=utf-8

import hashlib
import logging
import time
import tqdm

import tensorflow as tf

from envs.micro_grid import MicroGrid
from newa2c import Agent

if __name__ == "__main__":

    agent = Agent(128, 0, -200, 200, 100, 400, 0, 500, 0.0035, 0.0025, 8, 300)
    env = MicroGrid()
    env.define_observation_space('./envs/load.csv', './envs/prize.csv', './envs/pv.csv')

    first_state = tf.expand_dims(env.reset(), 0)
    with tqdm.trange(20) as t:
        for i in t:
            if i != 0:
                env.turn()
            for j in range(env.columns):
                next_state, reward = agent.multi_object(first_state, env)
                # print(reward)
                first_state = next_state
            t.set_description(f'Episode {i}')
    first_state = tf.expand_dims(env.reset(), 0)
    _, reward = agent.multi_object(first_state, env)
    print("test reward", reward)
