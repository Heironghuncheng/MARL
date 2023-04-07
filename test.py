# coding=utf-8

import tqdm
import gymnasium as gym
import tensorflow as tf
from envs.func import SecFunc
from newa2c_test import Agent

if __name__ == "__main__":
    turn = 3
    agent = Agent(128, 0, -200, 200, 100, 400, 0, 500, 0.0035, 0.0025, 8, 300)
    # env = SecFunc()
    writer = tf.summary.create_file_writer("./log")
    # first_state = tf.expand_dims(env.reset(), 0)
    env = gym.make('MountainCarContinuous-v0')
    # first_state = tf.expand_dims(env.reset(), 0)
    first_state, _ = env.reset()
    first_state = tf.expand_dims(first_state, 0)
    with tqdm.trange(1000) as t:
        for i in t:
            next_state, reward, action = agent.multi_object(first_state, env, writer, i, turn)
            t.set_description(f'Episode {i} state {next_state} reward {reward} action {action}')
            with writer.as_default():
                tf.summary.scalar('reward' + str(turn), float(reward), step=i)
                tf.summary.flush()
    writer.close()

