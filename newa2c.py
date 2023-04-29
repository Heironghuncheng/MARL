# coding=utf-8

import json
import logging
import os
import random
from time import time
from typing import Union

import messaging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (tf.Tensor,)):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class Actor(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=0.2)
        bias_initializer = tf.keras.initializers.Constant(0.1)
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer)
        self.normal1 = tf.keras.layers.LayerNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer)
        self.normal2 = tf.keras.layers.LayerNormalization()
        # , activation = "tanh"
        self.actor_pd_u = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer)
        self.actor_pd_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer)
        self.actor_pg_u = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer)
        self.actor_pg_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer)
        # self.actor_pb_u = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        # self.actor_pb_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=initializer)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.normal1(x)
        x = self.hidden2(x)
        x = self.normal2(x)
        #  (
        #             self.actor_pb_u(x), tf.math.exp(self.actor_pb_sig(x)))
        return (abs(self.actor_pd_u(x)), abs(self.actor_pd_sig(x))), (
            abs(self.actor_pg_u(x)), abs(self.actor_pg_sig(x)))


class Critic(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        bias_initializer = tf.keras.initializers.Constant(0.1)
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer)
        self.normal1 = tf.keras.layers.LayerNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer)
        self.normal2 = tf.keras.layers.LayerNormalization()
        self.critic = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer)

    def call(self, inputs, training=None, mask=None):
        # inputs = tf.reshape(inputs, shape=(1, 1))
        x = self.hidden1(inputs)
        x = tf.squeeze(x)
        x = self.normal1(x)
        x = tf.expand_dims(x, axis=0)
        x = self.hidden2(x)
        x = tf.squeeze(x)
        x = self.normal2(x)
        x = tf.expand_dims(x, axis=0)
        return self.critic(x)


class AveragedReturn(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=random.randint(1, 100))
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.normal1 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.normal2 = tf.keras.layers.BatchNormalization()
        self.returns = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer=initializer)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.normal1(x)
        x = self.hidden2(x)
        x = self.normal2(x)
        return self.returns(x)


def save_model(model: Union[Critic, Actor, AveragedReturn], exname=None, base="model"):
    base = "./" + base + "/"
    if exname is None:
        path = "model_" + str(type(model)).split("'")[1].split(".")[1]
    else:
        path = "model_" + str(type(model)).split("'")[1].split(".")[1] + exname
    if os.path.exists(base + path):
        pass
    else:
        os.mkdir(base + path)
    model.save_weights(base + path + "/ckpt")
    return


def load_model(model: Union[Critic, Actor, AveragedReturn], exname=None, base="model"):
    base = "./" + base + "/"
    if exname is None:
        path = "model_" + str(type(model)).split("'")[1].split(".")[1]
    else:
        path = "model_" + str(type(model)).split("'")[1].split(".")[1] + exname
    if os.path.exists(base + path):
        if isinstance(model, AveragedReturn):
            model(tf.expand_dims([0, 0, 0], 0))
        else:
            model(tf.expand_dims([0, 0, 0], 0))
        model.load_weights(base + path + "/ckpt")
    return model


def save_optimizer(optimizer: tf.keras.optimizers.Adam, name: str, base="model"):
    base = "./" + base + "/"
    path = "optimizer"
    if os.path.exists(base + path):
        pass
    else:
        os.mkdir(base + path)
    with open(base + 'optimizer/' + name + '_optimizer.json', 'w') as f:
        json.dump(optimizer.get_config(), f, cls=NumpyEncoder)
    return


def load_optimizer(optimizer: tf.keras.optimizers.Adam, name: str, base="model"):
    base = "./" + base + "/"
    if os.path.exists(base + 'optimizer/' + name + '_optimizer.json'):
        with open(base + 'optimizer/' + name + '_optimizer.json', 'r') as f:
            optimizer.from_config(json.load(f))
    return optimizer


def save_nums(nums: dict, base="model"):
    base = "./" + base + "/"
    path = "nums"
    if os.path.exists(base):
        pass
    else:
        os.mkdir(base)
    with open(base + path + '.json', 'w') as f:
        json.dump(nums, f, cls=NumpyEncoder)
    return


def load_nums(base="model"):
    base = "./" + base + "/"
    path = "nums"
    if os.path.exists(base + path + ".json"):
        with open(base + path + '.json', 'r') as f:
            return dict(json.load(f))
    else:
        return {
            "state_t": None,
            "state_t_plus": None,
            "value_t": {"env": 0., "money": 0.},
            "value_t_plus": {"env": 0., "money": 0.},
            "reward": {"env": 0., "money": 0.},
            "long_term_estimate": {"env": 0., "money": 0.},
            "avg_long_term_estimate": {"env": 0., "money": 0.},
            "action": None,
            "action_prob": None,
        }


def write(writer, turn, step, var, name):
    with writer.as_default():
        tf.summary.scalar(name + str(turn), var, step=step)
        tf.summary.flush()
    return


class Agent(object):
    def __init__(self, num_hidden_units: int, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                 costb, costc, voltage_para, base):

        self.actor = load_model(Actor(num_hidden_units))
        self.critic_env = load_model(Critic(num_hidden_units), "_env")
        self.critic_money = load_model(Critic(num_hidden_units), "_money")
        self.averaged_return_env = load_model(AveragedReturn(num_hidden_units), "_env")
        self.averaged_return_money = load_model(AveragedReturn(num_hidden_units), "_money")

        self.avg_long_optimizer_env = load_optimizer(tf.keras.optimizers.Adam(learning_rate=0.00001), "avg_return_env")
        self.avg_long_optimizer_money = load_optimizer(tf.keras.optimizers.Adam(learning_rate=0.00001),
                                                       "avg_return_money")
        self.actor_optimizer = load_optimizer(tf.keras.optimizers.Adam(learning_rate=0.00001), "actor")
        self.critic_env_optimizer = load_optimizer(tf.keras.optimizers.Adam(learning_rate=0.00001), "critic_env")
        self.critic_money_optimizer = load_optimizer(tf.keras.optimizers.Adam(learning_rate=0.00001), "critic_money")

        para = load_nums()
        # para
        self.state_t = para["state_t"]
        self.state_t_plus = para["state_t_plus"]
        self.value_t = para["value_t"]
        self.value_t_plus = para["value_t_plus"]
        self.reward = para["reward"]
        self.long_term_estimate = para["long_term_estimate"]
        self.avg_long_term_estimate = para["avg_long_term_estimate"]
        self.action = para["action"]
        self.action_prob = para["action_prob"]

        # special para
        self.soc = soc
        self.pb_min = pb_min
        self.pb_max = pb_max
        self.pd_min = pd_min
        self.pd_max = pd_max
        self.pg_min = pg_min
        self.pg_max = pg_max
        self.battery_cost = battery_cost
        self.costa = costa
        self.costb = costb
        self.costc = costc
        self.voltage_para = voltage_para
        self.base = base

        self.mes = messaging.Messaging(version="dev")
        filename = str(time()) + ".log"
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def param_dict(self):
        return {
            "state_t": self.state_t,
            "state_t_plus": self.state_t_plus,
            "value_t": self.value_t,
            "value_t_plus": self.value_t_plus,
            "reward": self.reward,
            "long_term_estimate": self.long_term_estimate,
            "avg_long_term_estimate": self.avg_long_term_estimate,
            "action": self.action,
            "action_prob": self.action_prob,
        }

    def alert(self, x):
        try:
            tf.debugging.assert_all_finite(x, "NAN OR INF")
        except Exception as e:
            self.mes.messaging(str(type(e)) + " " + str(e), self.param_dict(), "NAN OR INF")
            self.logger.error(("state_t", self.state_t))
            self.logger.error(("state_t_plus", self.state_t_plus))
            self.logger.error(("value_t", self.value_t))
            self.logger.error(("value_t_plus", self.value_t_plus))
            self.logger.error(("long_term_estimate", self.long_term_estimate))
            self.logger.error(("avg_long_term_estimate", self.avg_long_term_estimate))
            self.logger.error(("reward", self.reward))
            self.logger.error(("action", self.action))
            self.logger.error(("action_prob", self.action_prob))
            self.logger.error(x)
            raise

    def log(self, step):
        self.logger.info(("state_t", "step" + str(step), self.state_t))
        self.logger.info(("state_t_plus", "step" + str(step), self.state_t_plus))
        self.logger.info(("value_t", "step" + str(step), self.value_t))
        self.logger.info(("value_t_plus", "step" + str(step), self.value_t_plus))
        self.logger.info(("long_term_estimate", "step" + str(step), self.long_term_estimate))
        self.logger.info(("avg_long_term_estimate", "step" + str(step), self.avg_long_term_estimate))
        self.logger.info(("reward", "step" + str(step), self.reward))
        self.logger.info(("action", "step" + str(step), self.action))
        self.logger.info(("action_prob", "step" + str(step), self.action_prob))

    def save(self):
        save_model(self.actor)
        save_model(self.critic_env, "_env", self.base)
        save_model(self.critic_money, "_money", self.base)
        save_model(self.averaged_return_env, "_env", self.base)
        save_model(self.averaged_return_money, "_money", self.base)
        save_optimizer(self.actor_optimizer, "actor", self.base)
        save_optimizer(self.critic_env_optimizer, "critic_env", self.base)
        save_optimizer(self.critic_money_optimizer, "critic_money", self.base)
        save_optimizer(self.avg_long_optimizer_env, "avg_env", self.base)
        save_optimizer(self.avg_long_optimizer_money, "avg_money", self.base)
        save_nums(self.param_dict(), self.base)
        # save_long_return(self.long_term_estimate["env"], "long_env")
        # save_long_return(self.long_term_estimate["money"], "long_money")

    def communicate(self):
        pass

    def get_action(self):
        self.action_prob = self.actor(self.state_t)
        self.alert(self.action_prob)
        self.action = (
            tf.random.normal([1], mean=self.action_prob[0][0], stddev=self.action_prob[0][1]),
            tf.random.normal([1], mean=self.action_prob[1][0], stddev=self.action_prob[1][1]))
        return

    def long_term_func(self):
        self.long_term_estimate["env"] += 0.1 * (self.reward["env"] - self.long_term_estimate["env"])
        self.long_term_estimate["money"] += 0.1 * (self.reward["money"] - self.long_term_estimate["money"])
        return

    def value(self):
        with tf.GradientTape() as tape:
            self.value_t["env"] = self.critic_env(self.state_t)
            self.value_t_plus["env"] = self.critic_env(self.state_t_plus)
            self.alert(self.value_t_plus["env"])
            td_error = tf.reduce_mean(
                self.reward["env"] - self.long_term_estimate["env"] + self.value_t_plus["env"] - self.value_t[
                    "env"])
            loss = tf.square(td_error)
        grads = tape.gradient(loss, self.critic_env.trainable_variables)
        self.critic_env_optimizer.apply_gradients(zip(grads, self.critic_env.trainable_variables))
        with tf.GradientTape() as tape:
            self.value_t["money"] = self.critic_money(self.state_t)
            self.value_t_plus["money"] = self.critic_money(self.state_t_plus)
            self.alert(self.value_t_plus["money"])
            td_error = tf.reduce_mean(
                self.reward["money"] - self.long_term_estimate["money"] + self.value_t_plus["money"] - \
                self.value_t[
                    "money"])
            loss = tf.square(td_error)
        grads = tape.gradient(loss, self.critic_money.trainable_variables)
        self.critic_money_optimizer.apply_gradients(zip(grads, self.critic_money.trainable_variables))
        return

    def avg_long_term_func(self):
        # ls = [float(x) for x in
        #       [self.state_t[0][0], self.state_t[0][1], self.state_t[0][2], self.action[0][0], self.action[1][0]]]
        # inputs = tf.reshape(tf.constant(ls), [1, 5])
        with tf.GradientTape() as tape:
            self.avg_long_term_estimate["env"] = self.averaged_return_env(self.state_t)
            self.alert(self.avg_long_term_estimate["env"])
            loss = - tf.reduce_mean(self.reward["env"] - self.avg_long_term_estimate["env"])
            self.alert(loss)
        grads = tape.gradient(loss, self.averaged_return_env.trainable_variables)
        self.avg_long_optimizer_env.apply_gradients(zip(grads, self.averaged_return_env.trainable_variables))
        with tf.GradientTape() as tape:
            self.avg_long_term_estimate["money"] = self.averaged_return_money(self.state_t)
            self.alert(self.avg_long_term_estimate["money"])
            loss = - tf.reduce_mean(self.reward["money"] - self.avg_long_term_estimate["money"])
            self.alert(loss)
        grads = tape.gradient(loss, self.averaged_return_money.trainable_variables)
        self.avg_long_optimizer_money.apply_gradients(zip(grads, self.averaged_return_money.trainable_variables))
        return

    def op_act(self):
        td_error_env = self.avg_long_term_estimate["env"] - self.long_term_estimate["env"] + self.value_t_plus["env"] - \
                       self.value_t["env"]
        td_error_money = self.avg_long_term_estimate["money"] - self.long_term_estimate["money"] + self.value_t_plus[
            "money"] - self.value_t["money"]
        td_error = td_error_money + td_error_env
        dis = [tfp.distributions.Normal(loc=self.action_prob[i][0], scale=self.action_prob[i][1]) for i in range(2)]
        log_prob = tf.reduce_sum([dis[i].log_prob(value=self.action[i]) for i in range(2)])
        self.alert(log_prob)
        entropy = tf.reduce_sum([0.01 * i.entropy() for i in dis])
        loss = tf.multiply(log_prob, td_error) + entropy
        self.alert(loss)
        return loss

    def multi_object(self, init_state, env, writer, turn):
        all_reward = {"env": 0., "money": 0.}
        for step in range(24):
            with tf.GradientTape() as tape:
                self.state_t = init_state
                self.get_action()
                self.state_t_plus, self.reward["money"], self.reward["env"], self.soc = env.step(self.action,
                                                                                                 self.soc, self.pb_max,
                                                                                                 self.battery_cost,
                                                                                                 self.costa, self.costb,
                                                                                                 self.costc,
                                                                                                 self.voltage_para)
                self.long_term_func()
                self.avg_long_term_func()
                self.value()
                loss = self.op_act()
                self.alert(loss)
            grads = tape.gradient(loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            init_state = tf.reshape(self.state_t_plus, shape=(1, 3))
            all_reward["env"] += self.reward["env"]
            all_reward["money"] += self.reward["money"]
            step += 1
            self.log(step)
            # write(writer, turn, step, tf.squeeze(self.reward["env"]), "reward_env")
            # write(writer, turn, step, tf.squeeze(self.reward["money"]), "reward_money")
        self.save()
        return all_reward

    # def multi_agent(self, init_state, env):
    #     with tf.GradientTape() as tape:
    #         self.state_t = init_state
    #         self.get_action()
    #         action = list(self.action)
    #         if action[0] > self.pd_max:
    #             action[0] += self.pd_max - action[0]
    #         elif action[0] < self.pd_min:
    #             action[0] += self.pd_min - action[0]
    #         if action[1] > self.pg_max:
    #             action[1] += self.pg_max - action[1]
    #         elif action[1] < self.pg_min:
    #             action[1] += self.pg_min - action[1]
    #         # if action[2] > self.pb_max:
    #         #     action[2] += self.pb_max - action[2]
    #         # elif action[2] < self.pb_min:
    #         #     action[2] += self.pb_min - action[2]
    #         self.state_t_plus, self.reward["money"], self.reward["env"], self.soc = env.step(tuple(action),
    #                                                                                          self.soc, self.pb_max,
    #                                                                                          self.battery_cost,
    #                                                                                          self.costa, self.costb,
    #                                                                                          self.costc,
    #                                                                                          self.voltage_para)
    #         self.long_term_func()
    #         self.avg_long_term_func()
    #         self.value()
    #         loss = self.op_act()
    #         self.alert(loss)
    #     grads = tape.gradient(loss, self.actor.trainable_variables)
    #     self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    # write(writer, turn, step, tf.squeeze(self.reward), "reward")

# class DMOAC(object):
#     def __init__(self, url: str):
#         if os.path.exists(url):
#             with open(url, "r") as f:
#                 conf = json.load(f)
#         num_hidden_units = conf["num_hidden_units"]
#         self.agents = []
#         for agent in conf["agents"]:
#             self.agents.append(Agent(num_hidden_units, agent["soc"], agent["pb_min"], agent["pb_max"], agent["pd_min"],
#                                      agent["pd_max"],
#                                      agent["pg_min"], agent["pg_max"], agent["battery_cost"], agent["costa"],
#                                      agent["costb"],
#                                      agent["costc"], agent["voltage_para"]))
#         self.agents = tuple(self.agents)
#
#     def train(self, init_state, env):
#         pl = []
#         for agent in self.agents:
#             pl.append(Process(target=agent.multi_agent(init_state, env)))
#             pl[-1].start()
#         for process in pl:
#             process.join()
#
#
# if __name__ == "__main__":
#     pass
