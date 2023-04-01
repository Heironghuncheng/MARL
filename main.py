# coding=utf-8

import hashlib
import logging
import time

import tensorflow as tf

from envs.micro_grid import MicroGrid
from newa2c import Agent

if __name__ == "__main__":

    agent = Agent(128, 0, -200, 200, 100, 400, 0, 500, 0.0035, 0.0025, 8, 300)
    env = MicroGrid()
    env.define_observation_space('./envs/load.csv', './envs/prize.csv', './envs/pv.csv')

    first_state = tf.expand_dims(env.reset(), 0)
    for i in range(env.rows):
        if i != 0:
            env.turn()
        for t in range(env.columns):
            next_state, _ = agent.multi_object(first_state, env)
            print(_)
            first_state = next_state
    first_state = tf.expand_dims(env.reset(), 0)
    _, reward = agent.multi_object(first_state, env)
