# coding=utf-8

import tensorflow as tf

from envs.micro_grid import MicroGrid

from numpy import iterable
import logging
import time


class Actor(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units)
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units)
        self.actor_pd = tf.keras.layers.Dense(1, activation="relu")
        self.actor_pg = tf.keras.layers.Dense(1, activation="relu")
        self.actor_pb = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.actor_pd(x), self.actor_pg(x), self.actor_pb(x)


class Critic(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units)
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units)
        self.critic = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.critic(x)


class LongTermReturn(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units)
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units)
        self.returns = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.returns(x)


class AveragedReturn(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units)
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units)
        self.returns = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs: dict, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.returns(x)


def alert(x: tf.Tensor):
    x = tf.math.is_nan(x)
    if iterable(x):
        for i in x:
            if iterable(i):
                for j in i:
                    if iterable(j):
                        for n in j:
                            assert bool(n == tf.constant(False))
                    else:
                        assert bool(j == tf.constant(False))
            else:
                assert bool(i == tf.constant(False))
    else:
        assert bool(x == tf.constant(False))


class Agent:
    def __init__(self, num_hidden_units: int, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                 costb, costc):
        self.actor = Actor(num_hidden_units)
        self.critic = Critic(num_hidden_units)
        self.long_term_return = LongTermReturn(num_hidden_units)
        self.averaged_return = AveragedReturn(num_hidden_units)

        self.state_t = None
        self.state_t_plus = None
        self.value_t = None
        self.value_t_plus = None
        self.reward = None
        self.long_term_estimate = None
        self.avg_long_term_estimate = None
        self.action = None
        self.action_prob = None

        self.long_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.avg_long_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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

        logging.basicConfig(filename=str(time.time()) + ".log", filemode="w",
                            format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                            datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    def communicate(self):
        pass

    def get_action(self, state):
        alert(state)
        self.action_prob = self.actor(state)
        alert(self.action_prob)
        self.action = (
            tf.random.normal([1, 1], 0, self.action_prob[0]), tf.random.normal([1, 1], 0, self.action_prob[1]),
            tf.random.normal([1, 1], 0, self.action_prob[2]))
        return self.action

    def long_term_func(self, reward: tf.Tensor):
        with tf.GradientTape() as tape:
            alert(reward)
            long_term_estimate = self.long_term_return(reward)
            alert(long_term_estimate)
            loss = (long_term_estimate - reward) ** 2
            alert(loss)
        grads = tape.gradient(loss, self.long_term_return.trainable_variables)
        self.long_optimizer.apply_gradients(zip(grads, self.long_term_return.trainable_variables))
        return long_term_estimate

    def avg_long_term_func(self, reward: tf.Tensor):
        with tf.GradientTape() as tape:
            avg_long_term_estimate = self.averaged_return(reward)
            alert(avg_long_term_estimate)
            alpha = reward - avg_long_term_estimate
            alert(alpha)
            avg_long_term_return_ln = tf.math.log(avg_long_term_estimate)
            loss = -tf.math.reduce_sum(avg_long_term_return_ln * alpha)
            alert(loss)
        grads = tape.gradient(loss, self.averaged_return.trainable_variables)
        self.avg_long_optimizer.apply_gradients(zip(grads, self.averaged_return.trainable_variables))
        return avg_long_term_estimate

    def op_act(self, avg_long_term_return, long_term_return, state_t, state_t_plus, action_probs):
        if self.value_t_plus is None:
            value_t = self.critic(state_t)
        else:
            value_t = self.value_t_plus
        value_t_plus = self.critic(state_t_plus)
        td_error = avg_long_term_return - long_term_return + value_t_plus - value_t
        td_error = tf.reshape(td_error, [1, 1])
        alert(td_error)
        loss = None
        for i in action_probs:
            action_ln_probs = tf.math.log(i)
            alert(action_ln_probs)
            if loss is None:
                loss = -tf.math.reduce_sum(action_ln_probs * td_error)
            else:
                loss += -tf.math.reduce_sum(action_ln_probs * td_error)
        alert(loss)
        alert(value_t)
        alert(value_t_plus)
        return loss, value_t, value_t_plus

    @staticmethod
    def op_critic(long_term_return, value_t, value_t_plus, reward):
        td_error = reward - long_term_return + value_t_plus - value_t
        alert(td_error)
        value_t_plus_ln = tf.math.log(value_t_plus)
        print(value_t_plus)
        alert(value_t_plus_ln)
        loss = -tf.math.reduce_sum(value_t_plus_ln * td_error)

        return loss

    def multi_object(self, init_state, env: MicroGrid):
        with tf.GradientTape() as tape_:
            with tf.GradientTape() as tape:
                next_state, reward, self.soc = env.step(self.get_action(init_state), self.soc, self.pb_max,
                                                        self.battery_cost, self.costa,
                                                        self.costb, self.costc)
                self.state_t = init_state
                self.state_t_plus = next_state
                self.reward = reward
                self.long_term_estimate = self.long_term_func(self.reward)
                self.avg_long_term_estimate = self.avg_long_term_func(self.reward)
                logging.debug(("avg long term estimate", self.avg_long_term_estimate))
                logging.debug(("long term estimate", self.long_term_estimate))
                logging.debug(("reward", self.reward))
                logging.debug(("next state", self.state_t_plus))
                loss, self.value_t, self.value_t_plus = self.op_act(self.avg_long_term_estimate,
                                                                    self.long_term_estimate,
                                                                    self.state_t, self.state_t_plus, self.action_prob,
                                                                    )
                alert(loss)
                print("loss", loss)
            grads = tape.gradient(loss, self.actor.trainable_variables)
            logging.debug(("actor loss", loss))
            logging.debug(("actor grads", grads))
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            loss = self.op_critic(self.long_term_estimate, self.value_t, self.value_t_plus, reward)
            alert(loss)
        grads = tape_.gradient(loss, self.critic.trainable_variables)
        logging.debug(("critic loss", loss))
        logging.debug(("critic grads", grads))
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return next_state, reward

    def show(self, state):
        return self.get_action(state)


class DMOAC:
    def __init__(self, num_actions, num_hidden_units, agent_list, ):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    pass
