import numpy as np
import os
import glob
import matplotlib.pyplot as plt


load_path = "/media/qhawkins/Archive/MLM RL Model Logs/"

storage_dict = {}

step_list = []
combined_loss_list = []
actor_loss_list = []
critic_loss_list = []


for filename in glob.glob(load_path + "*.txt"):
    print(filename)
    day = filename.split("/")[-1].split(".")[0].split("rl_model_day_")[-1].split("_")[0]
    rank = filename.split("/")[-1].split(".")[0].split("rl_model_day_")[-1].split("_")[-1]
    storage_dict[day] = {}
    storage_dict[day][rank] = {}
    storage_dict[day][rank]["step"] = []
    with open(filename, "r") as f:
        data = f.readlines()
    for x in data:
        step = x.split("step: ")[-1].split(",")[0]
        print(step)
        combined_loss = x.split("combined_loss: ")[-1].split(",")[0]
        actor_loss = x.split("actor_loss: ")[-1].split(",")[0]
        critic_loss = x.split("critic_loss: ")[-1].split(",")[0]
        epsilon = x.split("epsilon: ")[-1].split(",")[0]
        accumulated_profit = x.split("accumulated_profit: ")[-1].split(",")[0]
        accumulated_step_reward = x.split("accumulated_step_reward: ")[-1].split(",")[0]
        accumulated_position = x.split("accumulated_position: ")[-1].split(",")[0]
        step_time = x.split("step_time: ")[-1].split(",")[0]
        accumulated_bh_profit = x.split("accumulated_bh_profit: ")[-1].split(",")[0]

        reward = x.split("reward: ")[-1].split(",")[0]
        
        step_list.append(step)
        combined_loss_list.append(combined_loss)
        actor_loss_list.append(actor_loss)
        critic_loss_list.append(critic_loss)
    plt.plot(range(len(combined_loss_list)), combined_loss_list)
    plt.title("Combined Loss")
    plt.show()
    plt.plot(range(len(combined_loss_list)), actor_loss_list)
    plt.title("Actor Loss")
    plt.show()
    plt.plot(range(len(combined_loss_list)), critic_loss_list)
    plt.title("Critic Loss")
    plt.show()

    #    exit()


    #storage_dict[day] = {}
    #storage_dict[day][rank] = 