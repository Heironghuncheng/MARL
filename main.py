# coding=utf-8

import json
import os
import random

import tensorflow as tf
import tqdm
from mpi4py import MPI

from envs.micro_grid import MicroGrid
from newa2c import MulAgent, Agent


def single_agent():
    writer = tf.summary.create_file_writer("./log")
    if os.path.exists("log"):
        pass
    else:
        os.mkdir("log")
    with open("running.txt", "w") as f:
        f.write(str(os.getpid()))
    all_base = "models/"
    agent = Agent(32, 0, -500, 500, 0, 1000, 0, 1000, 0.0035, 0.00025, 0.0025, 8, 0.05, all_base + "agent_1")
    env = MicroGrid(writer)
    env.define_observation_space('./envs/prize.csv', './envs/load.csv', './envs/pv.csv')
    with tqdm.trange(30000) as t:
        for i in t:
            first_state = env.reset(random.randint(0, 1369))
            first_state = tf.expand_dims(first_state, 0)
            reward = agent.single_agent(first_state, env)
            t.set_description(f'Episode {i} reward_env {float(reward["env"])} reward_money {float(reward["money"])}')
            with writer.as_default():
                tf.summary.scalar('all_reward_env', float(reward["env"]), step=i)
                tf.summary.scalar('all_reward_money', float(reward["money"]), step=i)
                tf.summary.flush()
        writer.close()
    with open("running.txt", "w") as f:
        f.write("finished")


def multi_agents():
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
    #     memory_limit=2048)])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    with open("conf.json", "r") as f:
        agents = json.load(f)
    writer = tf.summary.create_file_writer("./log")
    env = MicroGrid(writer)
    all_base = "models/"
    if 0 <= rank < 5:
        agent = MulAgent(agents["num_hidden_units"], agents["agents"][rank]["soc"],
                         agents["agents"][rank]["pb_min"], agents["agents"][rank]["pb_max"],
                         agents["agents"][rank]["pd_min"],
                         agents["agents"][rank]["pd_max"], agents["agents"][rank]["pg_min"],
                         agents["agents"][rank]["pg_max"], agents["agents"][rank]["battery_cost"],
                         agents["agents"][rank]["costa"],
                         agents["agents"][rank]["costb"], agents["agents"][rank]["costc"],
                         agents["agents"][rank]["voltage_para"], all_base + "agent_" + str(rank + 1), rank)
        env.define_observation_space('./envs/prize.csv', './envs/load.csv', './envs/pv.csv')
        print(f"rank {rank} created ")
    else:
        print("OUT OF RANK")
        assert False
    comm_ls = ([], [], [], [])
    with tqdm.trange(30000) as t:
        for i in t:
            first_state = env.reset(random.randint(0, 1369))
            first_state = tf.expand_dims(first_state, 0)
            reward = {"env": 0, "money": 0}
            for j in range(24):
                res = agent.multi_agent(first_state, env)
                reward["env"] += float(res["env"])
                reward["money"] += float(res["money"])
                for z in range(5):
                    comm_ls[0].append(comm.bcast(agent.critic_env.trainable_weights, root=z))
                    comm_ls[1].append(comm.bcast(agent.critic_money.trainable_weights, root=z))
                    comm_ls[2].append(comm.bcast(agent.averaged_return_env.trainable_weights, root=z))
                    comm_ls[3].append(comm.bcast(agent.averaged_return_money.trainable_weights, root=z))
                agent.communicate_cal(comm_ls)
                for z in comm_ls:
                    z.clear()
                first_state = tf.reshape(agent.state_t_plus, shape=(1, 3))
            t.set_description(
                f'Rank {rank} Episode {i} reward_env {float(reward["env"])} reward_money {float(reward["money"])}')
            with writer.as_default():
                tf.summary.scalar('all_reward_env', float(reward["env"]), step=i)
                tf.summary.scalar('all_reward_money', float(reward["money"]), step=i)
                tf.summary.flush()


if __name__ == "__main__":
    multi_agents()
