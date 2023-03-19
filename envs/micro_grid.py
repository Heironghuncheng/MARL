import numpy as np
from pandas import read_csv


class MicroGrid:

    def __init__(self):
        # action space
        # generate or buy or use battery
        self.__action_space_dict = {"P_DG_max": (-5,), "P_DG_min": (5,), "P_BT_max": (5,), "P_BT_min": (-5,),
                                    "P_PV_max": (-5,), "P_PV_min": (5,)}
        e = self.__action_space_dict
        self.action_space = np.array(
            [(e["P_DG_min"], e["P_DG_max"]), (e["P_PV_min"], e["P_PV_max"]), (e["P_BT_min"], e["P_BT_max"])],
            dtype=np.float16)

        # state
        self.state = None

        # observation space
        self.observation_space = None

        # node used to point data in the load-prize-pv table and the size of the table
        self.__node = [0, 0]
        self.columns = 0
        self.rows = 0

        # voltage
        self.__voltage_para = 1
        self.slack = 1

        # limitations
        # balance between supply and demand
        self.__more_balance_para = 3
        self.__less_balance_para = 5

        # battery capacity and battery
        self.__battery_cap = 100
        self.battery = 0
        self.__battery_para = 5
        self.__battery_dis = 10
        self.__battery_ch = 90
        # soc means last state of battery not now
        self.__soc = 0
        self.__raw_ch = 0
        self.__raw_dis = 0
        self.__last_battery_used = 0

        # voltage limit
        self.__low_voltage = 200
        self.__high_voltage = 240
        self.__voltage_limit = 5

        # target
        # Economic consumption function
        self.__generate_cost = 1
        self.__costa = 1
        self.__costb = 4
        self.__costc = 4

        # security
        self.__ref_voltage = 220

        # environmental objective
        self.__generate_env_param = 0.5
        self.__buy_env_param = 0.3

    @staticmethod
    def indicator(num):
        if num > 0:
            return 1
        else:
            return 0

    def step(self, action):
        # action: generate buy battery

        # voltage value
        p_dg, p_pv, p_bt = action
        v = p_dg * self.__voltage_para + self.slack

        # limitations

        # balance between supply and demand
        sup_demand_function = -(self.observation_space[1, self.__node[0], self.__node[1]] - self.observation_space[
            2, self.__node[0], self.__node[1]] - p_dg - p_pv + p_bt)
        if sup_demand_function > 0:
            sup_demand_punishment = self.__more_balance_para * sup_demand_function
        else:
            sup_demand_punishment = self.__less_balance_para * (- sup_demand_function)

        # battery
        self.battery -= p_bt
        self.__battery_dis = self.__soc * self.__battery_cap
        self.__battery_ch = (1 - self.__soc) * self.__battery_cap
        battery_punishment = max(self.__battery_dis - p_bt, 0) + max(self.__battery_ch - p_bt, 0)
        self.__soc += (self.indicator(self.__last_battery_used) * self.__raw_ch + (
                    1 - self.indicator(self.__last_battery_used)) / self.__raw_dis) * p_bt
        self.__last_battery_used = p_bt

        # voltage
        voltage_punishment = self.__voltage_limit * (max(v - self.__high_voltage, 0) + max(self.__low_voltage - v, 0))

        limitations = sup_demand_punishment + battery_punishment + voltage_punishment

        # target

        # Economic Object
        # maintenance of battery
        maintenance = self.__generate_cost * p_bt ** 2
        # generate costs
        generate_costs = self.__costa * p_dg ** 2 + self.__costb * p_dg + self.__costc
        # buy
        buy = p_pv * self.observation_space[0, self.__node[0], self.__node[1]]

        eco_reward = maintenance + generate_costs + buy

        # security objective
        sec_reward = (v - self.__ref_voltage) ** 2

        # environmental objective
        env_reward = p_dg * self.__generate_env_param + p_pv * self.__buy_env_param

        # reward
        reward = - eco_reward - sec_reward - env_reward - limitations

        # refresh state
        self.state = (self.observation_space[0, self.__node[0], self.__node[1]],
                      self.observation_space[1, self.__node[0], self.__node[1]],
                      self.observation_space[2, self.__node[0], self.__node[1]],
                      self.battery)

        # point to the next column or step or data
        self.__node[1] += 1

        return self.state, reward, False, False

    def turn(self):
        # next turn is beginning and the step reset to 0
        self.__node[0] += 1
        self.__node[1] = 0

        return self.__node[0]

    def reset(self):
        # first turn is beginning turn and step are reset to 0
        self.__node = [0, 0]
        self.battery = 0
        # refresh state
        self.state = (
            self.observation_space[0, 0, 0], self.observation_space[1, 0, 0], self.observation_space[2, 0, 0],
            self.battery)

        return self.state

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
