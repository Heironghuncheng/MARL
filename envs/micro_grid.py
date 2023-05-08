# coding=utf-8

import numpy as np
from pandas import read_csv
import tensorflow as tf


def write(writer, turn, step, var, name):
    with writer.as_default():
        tf.summary.scalar(name + str(turn), var, step=step)
        tf.summary.flush()


class MicroGrid:

    def __init__(self, writer):
        # observation space
        self.observation_space = None

        # node used to point data in the load-prize-pv table and the size of the table
        self.__node = [0, 0]
        self.columns = 0
        self.rows = 0

        # limitations
        # balance between supply and demand
        self.__more_balance_para = 1
        self.__less_balance_para = 1

        # battery capacity and battery
        # self.__battery_cap = 100
        # self.__battery_para = 5
        # self.__battery_dis = 10
        # self.__battery_ch = 90
        # soc means last state of battery not now
        # self.__soc = 0
        # self.__raw_ch = 0.98
        # self.__raw_dis = 0.98

        self.writer = writer
        self.time_step = 1

        # environmental objective
        self.__env_param = 1.6491

    def write(self, step, var, name):
        with self.writer.as_default():
            tf.summary.scalar(name, var, step=step)
            tf.summary.flush()

    @staticmethod
    def indicator(num):
        if num > 0:
            return 1
        else:
            return 0

    def step(self, action, soc, battery_cap, battery_cost, costa, costb, costc):

        # action: generate buy battery

        pd, pg = action

        # limitations

        # balance between supply and demand
        sup_demand_function = -(self.observation_space[1, self.__node[0], self.__node[1]] - self.observation_space[
            2, self.__node[0], self.__node[1]] - pd - pg)
        if sup_demand_function > 0:
            sup_demand_punishment = self.__more_balance_para * sup_demand_function
        else:
            sup_demand_punishment = self.__less_balance_para * - sup_demand_function

        # sup punish max = 13200

        # battery
        # print(soc)
        # battery_dis = soc * battery_cap
        # battery_ch = (1 - soc) * battery_cap
        # battery_punishment = 30 * (max(battery_dis - pb, 0) + max(battery_ch - pb, 0))
        # print(pb, battery_cap)
        # soc += (self.indicator(pb) * self.__raw_ch + (1 - self.indicator(pb)) / self.__raw_dis) * pb / battery_cap

        # battery max = 15000

        limitations = 10*sup_demand_punishment #0505   1->10

        # limit max = 45000

        # target

        # Economic Object
        # maintenance of battery
        # maintenance = battery_cost * pb ** 2
        # 875

        # generate costs
        generate_costs = costa * pd ** 2 + 0.1 * costb * pd + costc
        # 3500

        # buy
        buy = 0.05 * pg * self.observation_space[0, self.__node[0], self.__node[1]]
        # 5000

        eco_reward = generate_costs + buy
        # 10000

        # environmental objective
        env_reward = pd * self.__env_param + pg * self.__env_param
        # 3200

        # reward
        print("load", self.observation_space[1, self.__node[0], self.__node[1]])
        reward = (- eco_reward - env_reward) - 0.9*limitations
        print(limitations/sum(eco_reward,env_reward))
        # eco_reward = -eco_reward - limitations * 10
        # env_reward = - env_reward - limitations * 10
        reward = reward / 40 + 0.25     
        self.write(self.time_step, tf.squeeze(limitations), "limitations")
        self.time_step += 1

        # refresh state
        state = (self.observation_space[0, self.__node[0], self.__node[1]],
                 self.observation_space[1, self.__node[0], self.__node[1]],
                 self.observation_space[2, self.__node[0], self.__node[1]])

        # print(limitations)

        state = tf.expand_dims(state, 0)
        # eco_reward = tf.expand_dims(eco_reward, 0)
        # env_reward = tf.expand_dims(env_reward, 0)
        reward = tf.expand_dims(reward, 0)
        # print(env_reward, eco_reward)

        # point to the next column or step or data
        self.__node[1] += 1
        return state, reward, soc

    def reset(self, num: int = -1):
        # first turn is beginning turn and step are reset to 0
        if num >= 0:
            self.__node = [num if num <= 1369 else 1369, 0]
        else:
            self.__node = [self.__node[0] + 1 if self.__node[0] <= 1369 else 0, 0]
        # refresh state
        state = (
            self.observation_space[0, self.__node[0], 0], self.observation_space[1, self.__node[0], 0], self.observation_space[2, self.__node[0], 0],)

        return state

    def define_observation_space(self, prize_url="prize.csv", load_url="load.csv", pv_url="pv.csv"):
        # load the load, prize and pv data
        prize_data = np.array(read_csv(prize_url, header=None))
        load_data = np.array(read_csv(load_url, header=None))
        pv_data = np.array(read_csv(pv_url, header=None))

        # create a three three-dimensional array
        self.observation_space = np.array([prize_data, load_data, pv_data], dtype=np.float32)
        self.rows = prize_data.shape[0]
        self.columns = prize_data.shape[1]
        return self.rows, self.columns


if __name__ == "__main__":
    writer = tf.summary.create_file_writer("./log")
    env = MicroGrid(writer)
