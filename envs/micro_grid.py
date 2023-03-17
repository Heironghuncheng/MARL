import numpy as np
from pandas import read_csv


class MicroGrid:

    def __init__(self):
        # action space
        # generate or buy or use battery
        self.action_space = (-5, 5)

        # state
        self.state = None

        # observation space
        self.observation_space = None

        # node used to point data in the load-prize-pv table and the size of the table
        self.__node = [0, 0]
        self.columns = 0
        self.rows = 0

        # limitations
        # balance between supply and demand
        self.__more_balance_para = 3
        self.__less_balance_para = 5

        # battery capacity and battery
        self.__battery_cap = 100
        self.battery = 0
        self.__battery_para = 5

        # target
        # Economic consumption function
        self.__generate_cost = 1
        self.__costa = 1
        self.__costb = 4
        self.__costc = 4

        # security
        self.__security_para = 1
        self.slack = 1

        # environmental objective
        self.__generate_env_param = 0.5
        self.__buy_env_param = 0.3

    def step(self, action):
        # action: generate buy battery

        # limitations

        # balance between supply and demand
        sup_demand_function = -(self.observation_space[1, self.__node[0], self.__node[1]] - self.observation_space[
            2, self.__node[0], self.__node[1]] - action[0] - action[1] + action[2])
        if sup_demand_function > 0:
            sup_demand_punishment = self.__more_balance_para * sup_demand_function
        else:
            sup_demand_punishment = self.__less_balance_para * (- sup_demand_function)

        # battery
        self.battery -= action[2]

        if self.battery > self.__battery_cap:
            battery_punishment = self.__battery_para * (self.battery - self.__battery_cap)
        elif self.battery < 0:
            battery_punishment = self.__battery_para * (- self.battery)
        else:
            battery_punishment = 0

        limitations = sup_demand_punishment + battery_punishment

        # target

        # Economic Object
        # maintenance of battery
        maintenance = self.__generate_cost * action[2] ** 2
        # generate costs
        generate_costs = self.__costa * action[0] ** 2 + self.__costb * action[0] + self.__costc
        # buy
        buy = action[1] * self.observation_space[0, self.__node[0], self.__node[1]]

        eco_reward = maintenance + generate_costs + buy

        # security objective
        sec_reward = action[0] * self.__security_para + self.slack

        # environmental objective
        env_reward = action[0] * self.__generate_env_param + action[1] * self.__buy_env_param

        # reward
        reward = - eco_reward - sec_reward - env_reward - limitations

        # refresh state
        self.state = (self.observation_space[0, self.__node[0], self.__node[1]],
                      self.observation_space[1, self.__node[0], self.__node[1]],
                      self.observation_space[2, self.__node[0], self.__node[1]],
                      self.battery)

        # point to the next column or step or data
        self.__node[1] += 1

        return self.state, reward, False, False, {}

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
        self.observation_space[0, 0, 0], self.observation_space[1, 0, 0], self.observation_space[2, 0, 0], self.battery)

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
