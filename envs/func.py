# coding=utf-8

import numpy as np
from pandas import read_csv
import tensorflow as tf


class SecFunc:
    def __init__(self):
        self.__a = -4.
        self.__b = 4.
        self.__c = 1000.
        self.state = 0.

    def step(self, action):
        next_state = self.state + action
        if next_state < -10.:
            next_state = tf.constant(-10., dtype="float32", shape=(1, 1, 1))
        elif next_state > 20.:
            next_state = tf.constant(20., dtype="float32", shape=(1, 1, 1))
        reward = self.__a * next_state * next_state + self.__b * next_state + self.__c
        self.state = next_state
        return tf.expand_dims(next_state, 0), tf.expand_dims(reward, 0),

    def reset(self):
        self.state = 0.
        return tf.expand_dims(self.state, 0)


