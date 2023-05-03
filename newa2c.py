# coding=utf-8

import json
import logging
import os
import random
import time
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import messaging


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

    def call(self, inputs, training=None, mask=None) -> tuple:
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

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
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
        self.normal1 = tf.keras.layers.LayerNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.normal2 = tf.keras.layers.LayerNormalization()
        self.returns = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer=initializer)

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        x = self.hidden1(inputs)
        x = self.normal1(x)
        x = self.hidden2(x)
        x = self.normal2(x)
        return self.returns(x)


def write(writer, turn, step, var, name):
    with writer.as_default():
        tf.summary.scalar(name + str(turn), var, step=step)
        tf.summary.flush()
    return


class Agent(object):
    def __init__(self, num_hidden_units: int, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                 costb, costc, voltage_para, base):

        self.base = "./" + base + "/"
        self.path_ls = (
            "actor", "critic_env", "critic_money", "averaged_return_env", "averaged_return_money", "actor_optimizer",
            "critic_env_optimizer", "critic_money_optimizer", "avg_long_optimizer_env", "avg_long_optimizer_money",
            "vars")

        self.actor = Actor(num_hidden_units)
        self.critic_env = Critic(num_hidden_units)
        self.critic_money = Critic(num_hidden_units)
        self.averaged_return_env = AveragedReturn(num_hidden_units)
        self.averaged_return_money = AveragedReturn(num_hidden_units)

        self.avg_long_optimizer_env = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.avg_long_optimizer_money = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.critic_env_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.critic_money_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

        self.load()

        # para
        para = self.load_nums()
        self.state_t = para["state_t"]
        self.state_t_plus = para["state_t_plus"]
        self.value_t = para["value_t"]
        self.value_t_plus = para["value_t_plus"]
        self.reward = para["reward"]
        self.long_term_estimate = para["long_term_estimate"]
        self.avg_long_term_estimate = para["avg_long_term_estimate"]
        self.action = para["action"]
        self.action_prob = para["action_prob"]

        for i in range(5):
            if os.path.exists(self.base + self.path_ls[i]):
                pass
            else:
                os.makedirs(self.base + self.path_ls[i])

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

        self.mes = messaging.Messaging(version="dev")
        filename = str(time.time()) + ".log"
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def save(self):
        self.actor.save_weights(self.base + self.path_ls[0] + "/ckpt")
        self.critic_env.save_weights(self.base + self.path_ls[1] + "/ckpt")
        self.critic_money.save_weights(self.base + self.path_ls[2] + "/ckpt")
        self.averaged_return_env.save_weights(self.base + self.path_ls[3] + "/ckpt")
        self.averaged_return_money.save_weights(self.base + self.path_ls[4] + "/ckpt")
        with open(self.base + self.path_ls[5] + '.json', 'w') as f:
            json.dump(self.actor_optimizer.get_config(), f, cls=NumpyEncoder)
        with open(self.base + self.path_ls[6] + '.json', 'w') as f:
            json.dump(self.critic_env_optimizer.get_config(), f, cls=NumpyEncoder)
        with open(self.base + self.path_ls[7] + '.json', 'w') as f:
            json.dump(self.critic_money_optimizer.get_config(), f, cls=NumpyEncoder)
        with open(self.base + self.path_ls[8] + '.json', 'w') as f:
            json.dump(self.avg_long_optimizer_env.get_config(), f, cls=NumpyEncoder)
        with open(self.base + self.path_ls[9] + '.json', 'w') as f:
            json.dump(self.avg_long_optimizer_money.get_config(), f, cls=NumpyEncoder)
        with open(self.base + self.path_ls[10] + '.json', 'w') as f:
            json.dump(self.param_dict(), f, cls=NumpyEncoder)

    def load(self):
        if os.path.exists(self.base + self.path_ls[0]):
            self.actor(tf.expand_dims([0, 0, 0], 0))
            self.actor.load_weights(self.base + self.path_ls[0] + "/ckpt")
            self.critic_env(tf.expand_dims([0, 0, 0], 0))
            self.critic_env.load_weights(self.base + self.path_ls[1] + "/ckpt")
            self.critic_money(tf.expand_dims([0, 0, 0], 0))
            self.critic_money.load_weights(self.base + self.path_ls[2] + "/ckpt")
            self.averaged_return_env(tf.expand_dims([0, 0, 0], 0))
            self.averaged_return_env.load_weights(self.base + self.path_ls[3] + "/ckpt")
            self.averaged_return_money(tf.expand_dims([0, 0, 0], 0))
            self.averaged_return_money.load_weights(self.base + self.path_ls[4] + "/ckpt")
            with open(self.base + self.path_ls[5] + '.json', 'r') as f:
                self.actor_optimizer.from_config(json.load(f))
            with open(self.base + self.path_ls[6] + '.json', 'r') as f:
                self.critic_env_optimizer.from_config(json.load(f))
            with open(self.base + self.path_ls[7] + '.json', 'r') as f:
                self.critic_money_optimizer.from_config(json.load(f))
            with open(self.base + self.path_ls[8] + '.json', 'r') as f:
                self.avg_long_optimizer_env.from_config(json.load(f))
            with open(self.base + self.path_ls[9] + '.json', 'r') as f:
                self.avg_long_optimizer_money.from_config(json.load(f))
        else:
            pass

    def load_nums(self):
        if os.path.exists(self.base + self.path_ls[10] + '.json'):
            with open(self.base + self.path_ls[10] + '.json', 'r') as f:
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

    def param_dict(self) -> dict:
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
            with open("running.txt", "w") as f:
                f.write("NAN OR INF")
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

    def op_act(self) -> tf.Tensor:
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

    def single_step(self, init_state, env):
        with tf.GradientTape() as tape:
            self.state_t = init_state
            self.get_action()
            # sig = True
            # while True:
            #     stop = yield sig
            #     if not stop:
            #         break
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

    def single_agent(self, init_state, env) -> dict:
        # , writer, turn
        all_reward = {"env": 0., "money": 0.}
        for step in range(24):
            self.single_step(init_state, env)
            init_state = tf.reshape(self.state_t_plus, shape=(1, 3))
            all_reward["env"] += self.reward["env"]
            all_reward["money"] += self.reward["money"]
            step += 1
            self.log(step)
            # write(writer, turn, step, tf.squeeze(self.reward["env"]), "reward_env")
            # write(writer, turn, step, tf.squeeze(self.reward["money"]), "reward_money")
        self.save()
        return all_reward


class MulAgent(Agent):
    def __init__(self, num_hidden_units: int, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                 costb, costc, voltage_para, base):
        super().__init__(num_hidden_units, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                         costb, costc, voltage_para, base)
        self.__step = 0
        self.comm_map = ((0.41667, 0, 0.33333, 0.25, 0),
                         (0, 0.41667, 0.33333, 0.25, 0),
                         (0.33333, 0.33333, 0.33334, 0, 0),
                         (0.25, 0.25, 0, 0.25, 0.25),
                         (0, 0, 0, 0.25, 0.75))

    def next_step(self):
        self.__step += 1

    def param_dict(self) -> dict:
        return {
            "value_t": self.value_t,
            "value_t_plus": self.value_t_plus,
            "reward": self.reward,
            "long_term_estimate": self.long_term_estimate,
            "avg_long_term_estimate": self.avg_long_term_estimate,
            "step": self.__step,
        }

    def communicate_cal(self, data: tuple[list, list, list, list], rank):
        self.critic_env.set_weights(sum([self.comm_map[rank][i] * data[0][i] for i in range(5)]))
        self.critic_money.set_weights(sum([self.comm_map[rank][i] * data[1][i] for i in range(5)]))
        self.averaged_return_env.set_weights(sum([self.comm_map[rank][i] * data[2][i] for i in range(5)]))
        self.averaged_return_money.set_weights(sum([self.comm_map[rank][i] * data[3][i] for i in range(5)]))

    def communicate_give(self):
        base = self.base + "/"
        if os.path.exists(base):
            pass
        else:
            os.mkdir(base)
        with open(base + "data.json", "w") as f:
            json.dump(self.param_dict(), f, cls=NumpyEncoder)

    def communicate_receive(self, url: str):
        url += "/"
        if os.path.exists(url):
            with open(url + "data.json", "r") as f:
                data = dict(json.load(f))
            if data["step"] == self.__step:
                self.communicate_cal(data)
                return True
        return False

    def multi_agent(self, init_state, env):
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
        return self.reward

