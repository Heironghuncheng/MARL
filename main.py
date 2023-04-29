# coding=utf-8

import os
import tqdm
import tensorflow as tf
from envs.micro_grid import MicroGrid
from newa2c import Agent
import random

if __name__ == "__main__":
    writer = tf.summary.create_file_writer("./log")
    if os.path.exists("model"):
        pass
    else:
        os.mkdir("model")
    agent = Agent(32, 0, -500, 500, 0, 1000, 0, 1000, 0.0035, 0.00025, 0.0025, 8, 0.05, "model")
    env = MicroGrid(writer)
    env.define_observation_space('./envs/prize.csv', './envs/load.csv', './envs/pv.csv')
    with tqdm.trange(30000) as t:
        for i in t:
            first_state = env.reset(random.randint(0, 1369))
            first_state = tf.expand_dims(first_state, 0)
            reward = agent.multi_object(first_state, env, writer, i)
            t.set_description(f'Episode {i} reward_env {float(reward["env"])} reward_money {float(reward["money"])}')
            with writer.as_default():
                tf.summary.scalar('all_reward_env', float(reward["env"]), step=i)
                tf.summary.scalar('all_reward_money', float(reward["money"]), step=i)
                tf.summary.flush()
        writer.close()

