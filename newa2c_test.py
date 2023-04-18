# coding=utf-8

import logging
import random
import tensorflow as tf
import tensorflow_probability as tfp
import time
import winsound


class Actor(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.1, seed=random.randint(1, 100))
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="relu", kernel_initializer=initializer)
        self.normal1 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="relu", kernel_initializer=initializer)
        self.normal2 = tf.keras.layers.BatchNormalization()
        # , activation = "tanh"
        self.actor_pd_u = tf.keras.layers.Dense(1, activation="relu", kernel_initializer=initializer)
        self.actor_pd_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=initializer)
        self.actor_pg_u = tf.keras.layers.Dense(1, activation="relu", kernel_initializer=initializer)
        self.actor_pg_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=initializer)
        self.actor_pb_u = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        self.actor_pb_sig = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=initializer)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.normal1(x)
        x = self.hidden2(x)
        x = self.normal2(x)
        #  (
        #             self.actor_pb_u(x), tf.math.exp(self.actor_pb_sig(x)))
        return (self.actor_pd_u(x), tf.math.exp(self.actor_pd_sig(x))), (self.actor_pg_u(x), tf.math.exp(self.actor_pg_sig(x)))


class Critic(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=random.randint(1, 100))
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=initializer)
        self.normal1 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="tanh", kernel_initializer=initializer)
        self.normal2 = tf.keras.layers.BatchNormalization()
        self.critic = tf.keras.layers.Dense(1, kernel_initializer=initializer)

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


# class LongTermReturn(tf.keras.Model):
#     def __init__(self, num_hidden_units: int):
#         super().__init__()
#         self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
#         self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
#         self.returns = tf.keras.layers.Dense(1)
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.hidden1(inputs)
#         x = self.hidden2(x)
#         return self.returns(x)


class AveragedReturn(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.hidden2 = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.returns = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.returns(x)


def write(writer, turn, step, var, name):
    with writer.as_default():
        tf.summary.scalar(name + str(turn), var, step=step)
        tf.summary.flush()


class Agent:
    def __init__(self, num_hidden_units: int, soc, pb_min, pb_max, pd_min, pd_max, pg_min, pg_max, battery_cost, costa,
                 costb, costc, voltage_para):
        self.actor = Actor(num_hidden_units)
        self.critic = Critic(num_hidden_units)
        # self.long_term_return = LongTermReturn(num_hidden_units)
        self.averaged_return = AveragedReturn(num_hidden_units)

        self.state_t = None
        self.state_t_plus = None
        self.value_t = None
        self.value_t_plus = None
        self.reward = None
        self.long_term_estimate = 0.
        self.avg_long_term_estimate = 0.
        self.action = None
        self.action_prob = None
        self.action_goose = None

        # self.long_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.avg_long_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
        self.voltage_para = voltage_para

        logging.basicConfig(filename=str(time.time()) + ".log", filemode="w",
                            format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                            datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    def alert(self, x: tf.Tensor):
        # Check if x contains any NaN values
        try:
            tf.debugging.assert_all_finite(x, "Tensor contains NaN values")
        except:
            duration = 1000  # millisecond
            freq = 440  # Hz
            winsound.Beep(freq, duration)
            logging.error(self.state_t)
            logging.error(self.state_t_plus)
            logging.error(self.value_t)
            logging.error(self.value_t_plus)
            logging.error(self.reward)
            logging.error(self.action)
            logging.error(self.action_prob)
            logging.error(self.critic.trainable_variables)
            logging.error(x)
            tf.debugging.assert_all_finite(x, "Tensor contains NaN values")

    def communicate(self):
        pass

    def get_action(self, state):
        # self.alert(state)
        self.action_prob = self.actor(state)
        self.alert(self.action_prob)
        self.action = (
            tf.random.normal([1], mean=self.action_prob[0][0], stddev=self.action_prob[0][1]),
            tf.random.normal([1], self.action_prob[1][0], self.action_prob[1][1]))
        # ,
        #             tf.random.normal([1], self.action_goose[2][0], self.action_goose[2][1])
        # print(self.action_goose)
        # print("\n*****\n")
        # print(self.action)
        return self.action

    # @tf.function
    # def long_term_func(self, reward: tf.Tensor):
    #     self.alert(reward)
    #     with tf.GradientTape() as tape:
    #         long_term_estimate = self.long_term_return(reward)
    #         print(reward)
    #         self.alert(long_term_estimate)
    #         loss = - tf.square(long_term_estimate - reward)
    #         # print(loss)
    #         self.alert(loss)
    #     grads = tape.gradient(loss, self.long_term_return.trainable_variables)
    #     self.long_optimizer.apply_gradients(zip(grads, self.long_term_return.trainable_variables))
    #     # print(long_term_estimate)
    #     return long_term_estimate

    def long_term_func(self, reward):
        self.long_term_estimate += 0.01 * (reward - self.long_term_estimate)
        return self.long_term_estimate

    def avg_long_term_func(self, reward: tf.Tensor):
        with tf.GradientTape() as tape:
            avg_long_term_estimate = self.averaged_return(reward)
            self.alert(avg_long_term_estimate)
            loss = reward - avg_long_term_estimate
            self.alert(loss)
        grads = tape.gradient(loss, self.averaged_return.trainable_variables)
        self.avg_long_optimizer.apply_gradients(zip(grads, self.averaged_return.trainable_variables))
        return avg_long_term_estimate, loss

    def op_act(self, avg_long_term_return, long_term_return, state_t, state_t_plus, action_probs, reward):
        value_t = self.critic(state_t) if self.value_t_plus is None else self.value_t_plus
        value_t_plus = self.critic(state_t_plus)
        self.alert(value_t_plus)
        # td_error = avg_long_term_return - long_term_return + value_t_plus - value_t
        td_error = reward + 0.95 * value_t_plus - value_t
        # loss of actor must be positive
        loss = tf.math.reduce_sum([
            tf.multiply(tfp.distributions.Normal(loc=action_probs[i][0], scale=action_probs[i][1]).log_prob(
                value=self.action[i]), td_error) for i in range(2)])
        self.alert(action_probs)
        self.alert(td_error)
        self.alert(loss)
        return loss, value_t, value_t_plus

    @staticmethod
    def op_critic(long_term_return, value_t, value_t_plus, reward):
        td_error = reward - long_term_return + value_t_plus - value_t
        # td_error = value_t_plus - value_t
        # self.self.alert(td_error)
        # loss = -tf.reduce_sum(tf.losses.mean_squared_error(value_t_plus, reward + 0.95 * value_t))
        loss = -tf.square(td_error)
        # loss = -tf.reduce_sum(loss)
        return loss

    def multi_object(self, init_state, env, writer, turn):
        all_reward = 0
        for step in range(24):
            with tf.GradientTape() as tape_:
                with tf.GradientTape() as tape:
                    action = list(self.get_action(init_state))
                    if action[0] > self.pd_max:
                        action[0] += self.pd_max - action[0]
                    elif action[0] < self.pd_min:
                        action[0] += self.pd_min - action[0]
                    if action[1] > self.pg_max:
                        action[1] += self.pg_max - action[1]
                    elif action[1] < self.pg_min:
                        action[1] += self.pg_min - action[1]
                    # if action[2] > self.pb_max:
                    #     action[2] += self.pb_max - action[2]
                    # elif action[2] < self.pb_min:
                    #     action[2] += self.pb_min - action[2]
                    # write(writer, turn, step, tf.squeeze(action), "action")
                    next_state, reward, self.soc = env.step(tuple(action), self.soc, self.pb_max, self.battery_cost,
                                                            self.costa, self.costb, self.costc, self.voltage_para)
                    logging.info(reward)
                    logging.info(self.value_t_plus)
                    logging.info(self.action)
                    self.state_t = init_state
                    self.state_t_plus = next_state
                    self.reward = reward
                    self.long_term_estimate = self.long_term_func(self.reward)
                    self.avg_long_term_estimate, loss = self.avg_long_term_func(self.reward)
                    # write(writer, turn, step, loss, "loss_avg")
                    loss, self.value_t, self.value_t_plus = self.op_act(self.avg_long_term_estimate,
                                                                        self.long_term_estimate,
                                                                        self.state_t, self.state_t_plus,
                                                                        self.action_prob,
                                                                        reward
                                                                        )
                    self.alert(loss)
                grads = tape.gradient(loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                loss = self.op_critic(self.long_term_estimate, self.value_t, self.value_t_plus, self.reward)
            grads = tape_.gradient(loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            self.alert(loss)
            init_state = tf.reshape(next_state, shape=(1, 3))
            all_reward += reward
            step += 1
            write(writer, turn, step, tf.squeeze(reward), "reward")
            # if end:
            #     # write(writer, turn, step, all_reward, "reward")
            #     return all_reward
            # write(writer, turn, step, self.action_goose[0][1].nparray, "squre")
        return all_reward


class DMOAC:
    def __init__(self, num_actions, num_hidden_units, agent_list, ):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    pass
