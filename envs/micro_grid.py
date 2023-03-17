import numpy as np
from pandas import read_csv
from pprint import pprint


class MicroGrid:

    def __init__(self):
        self.__costa = 1
        self.__costb = 4
        self.__costc = 4
        self.__min_action = -2
        self.__max_action = 2
        self.__penalty_para = 5

        self.__node = None
        self.action_space = (-5, 5)
        self.observation_space = None
        self.state = None
        self.columns = None
        self.rows = None

    def action_space_func(self, num):
        if num < self.action_space[0]:
            num = self.action_space[0]
        elif num > self.action_space[1]:
            num = self.action_space[1]
        return num

    def step(self, action):
        ec_function = self.__costa * self.action_space_func(action) ** 2 + self.__costb * action + self.__costc
        reward = -ec_function - self.__penalty_para * abs(
            self.observation_space[1, self.__node[0], self.__node[1]] - self.observation_space[2, self.__node[0], self.__node[1]] - self.action_space_func(action)) - \
                 self.observation_space[0, self.__node[0], self.__node[1]]

        self.state = (self.observation_space[0, self.__node[0], self.__node[1]],
                      self.observation_space[1, self.__node[0], self.__node[1]],
                      self.observation_space[2, self.__node[0], self.__node[1]])
        self.__node[1] += 1
        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def turn(self):
        self.__node[0] += 1
        self.__node[1] = 0
        return

    def reset(self):
        self.__node = [0, 0]
        self.state = (self.observation_space[0, 0, 0], self.observation_space[1, 0, 0], self.observation_space[2, 0, 0])
        return

    def define_observation_space(self, prize_url="prize.csv", pv_url="pv.csv", load_url="load.csv"):
        prize_data = np.array(read_csv(prize_url, header=None))
        load_data = np.array(read_csv(load_url, header=None))
        pv_data = np.array(read_csv(pv_url, header=None))
        self.observation_space = np.array([prize_data, load_data, pv_data], dtype=np.float32)
        self.rows = prize_data.shape[0]
        self.columns = prize_data.shape[1]
