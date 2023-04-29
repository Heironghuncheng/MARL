# coding=utf-8

import os
import tqdm
import gymnasium as gym
import tensorflow as tf
# from envs.func import SecFunc
from envs.micro_grid import MicroGrid
from newa2c import Agent
import random

if __name__ == "__main__":
    # turn = 0
    writer = tf.summary.create_file_writer("./log")
    agent = Agent(128, 0, -500, 500, 0, 1000, 0, 1000, 0.0035, 0.00025, 0.0025, 8, 0.05)
    with open("running.txt", "w") as f:
        f.write(str(os.getpid()))
    env = MicroGrid(writer)
    env.define_observation_space('./envs/prize.csv', './envs/load.csv', './envs/pv.csv')
    with tqdm.trange(10000) as t:
        # env.render() random.randint(0, 1369)
        for i in t:
            first_state = env.reset(0)
            first_state = tf.expand_dims(first_state, 0)
            reward = agent.multi_object(first_state, env, writer, i)
            t.set_description(f'Episode {i} reward {reward}')
            with writer.as_default():
                tf.summary.scalar('all_reward', float(reward), step=i)
                tf.summary.flush()
        writer.close()
    with open("running.txt", "w") as f:
        f.write("finished")

