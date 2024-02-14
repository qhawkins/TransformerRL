import numpy as np
import os
import glob
import matplotlib.pyplot as plt


load_path = "/media/qhawkins/Archive/MLM RL Model Logs/"

storage_dict = {}

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
        combined_loss = x.split("combined_loss: ")[-1].split(",")[0]
        actor_loss = x.split("actor_loss: ")[-1].split(",")[0]
        critic_loss = x.split("critic_loss: ")[-1].split(",")[0]
        epsilon = x.split("epsilon: ")[-1].split(",")[0]
        accumulated_profit = x.split("accumulated_profit: ")[-1].split(",")[0]
        accumulated_step_reward = x.split("accumulated_step_reward: ")[-1].split(",")[0]
        
        reward = x.split("reward: ")[-1].split(",")[0]
        
        
        exit()


    #storage_dict[day] = {}
    #storage_dict[day][rank] = 