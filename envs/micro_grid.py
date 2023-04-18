# coding=utf-8

import numpy as np
from pandas import read_csv
import tensorflow as tf


class MicroGrid:

    def __init__(self, **config):
        # action space
        # generate or buy or use battery
        # self.__action_space_dict = {"PD_min": 100, "PD_max": 400, "PG_max": 500, "PG_min": 500,
        #                             "PB_min": -200, "PB_max": 200}
        # e = self.__action_space_dict
        # self.action_space = np.array(
        #     [(e["PD_min"], e["PD_max"]), (e["PG_min"], e["PG_max"]), (e["PB_min"], e["PB_max"])],
        #     dtype=np.float16)

        # state
        # self.state = None

        # observation space
        self.observation_space = None

        # node used to point data in the load-prize-pv table and the size of the table
        self.__node = [0, 0]
        self.columns = 0
        self.rows = 0

        # voltage
        # self.voltage_para = 30
        self.__slack = 220

        # limitations
        # balance between supply and demand
        self.__more_balance_para = 88
        self.__less_balance_para = 88

        # battery capacity and battery
        # self.__battery_cap = 100
        # self.__battery_para = 5
        # self.__battery_dis = 10
        # self.__battery_ch = 90
        # soc means last state of battery not now
        # self.__soc = 0
        self.__raw_ch = 0.98
        self.__raw_dis = 0.98

        # voltage limit
        self.__low_voltage = 213.4
        self.__high_voltage = 242
        self.__voltage_limit = 500

        # target
        # Economic consumption function
        # self.__battery_cost = 0.0035
        # self.__costa = 0.0025
        # self.__costb = 8
        # self.__costc = 300

        # security
        self.__ref_voltage = 220

        # environmental objective
        self.__env_param = 1.6491

    @staticmethod
    def indicator(num):
        if num > 0:
            return 1
        else:
            return 0

    def step(self, action, soc, battery_cap, battery_cost, costa, costb, costc, voltage_para):

        # action: generate buy battery

        # voltage value
        pd, pg= action
        # , pb
        v = pd * voltage_para + self.__slack

        # vmax = 270

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

        # voltage
        voltage_punishment = self.__voltage_limit * (max(v - self.__high_voltage, 0) + max(self.__low_voltage - v, 0))

        # voltage max = 15000
        limitations = sup_demand_punishment + voltage_punishment

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

        # security objective
        sec_reward = (v - self.__ref_voltage) ** 2
        # 900

        # environmental objective
        env_reward = pd * self.__env_param + pg * self.__env_param
        # 3200

        # reward
        reward = (- eco_reward - sec_reward - env_reward) - limitations * 10

        # refresh state
        state = (self.observation_space[0, self.__node[0], self.__node[1]],
                 self.observation_space[1, self.__node[0], self.__node[1]],
                 self.observation_space[2, self.__node[0], self.__node[1]])

        # print(limitations)


        state = tf.expand_dims(state, 0)
        reward = tf.expand_dims(reward, 0)

        # point to the next column or step or data
        self.__node[1] += 1

        return state, reward, soc

    def turn(self):
        # next turn is beginning and the step reset to 0
        self.__node[0] += 1
        self.__node[1] = 0

        return self.__node[0]

    def reset(self, num):
        # first turn is beginning turn and step are reset to 0
        if num > 1369:
            num = 1369
        self.__node = [1369, 0]
        # refresh state
        state = (
            self.observation_space[0, 0, 0], self.observation_space[1, 0, 0], self.observation_space[2, 0, 0],)

        return state

    def define_observation_space(self, prize_url="prize.csv", pv_url="pv.csv", load_url="load.csv"):
        # load the load, prize and pv data
        prize_data = np.array(read_csv(prize_url, header=None))
        load_data = np.array(read_csv(load_url, header=None))
        pv_data = np.array(read_csv(pv_url, header=None))

        # create a three three-dimensional array
        self.observation_space = np.array([prize_data, load_data, pv_data], dtype=np.float32)
        self.rows = prize_data.shape[0]
        self.columns = prize_data.shape[1]
        return self.rows, self.columns
