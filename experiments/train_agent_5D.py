#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:39:08 2020

@author: alan
"""
from callbacks import SaveOnBestTrainingReward
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2
import sys
from pathlib import Path


if __name__ == "__main__":
    n_envs = sys.argv[1]
    n_steps = sys.argv[2]
    log_dir = "/home/alan/Documents/RL-Models/Paper/PPO_5D_{0}_{1}".format(n_envs, n_steps)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(log_dir)
    print("Training with: {0} parallel agents, updating every: {1} timesteps".format(n_envs, n_steps))
    save_callback = SaveOnBestTrainingReward(check_freq=1000, log_dir=log_dir)
    vec_env = make_vec_env("path_following:Tool5D-v0", n_envs=int(n_envs), monitor_dir=log_dir)
    model = PPO2(MlpPolicy, vec_env, verbose=0, n_steps=int(n_steps))
    model.learn(total_timesteps=1000000, callback=save_callback)
    print("Training finished")
