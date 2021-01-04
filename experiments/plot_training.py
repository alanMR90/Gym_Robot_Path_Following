#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:34:31 2020

@author: alan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from stable_baselines.results_plotter import load_results, ts2xy


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'same')


updates = [8, 16, 32, 64]
agents = [4, 8, 16, 32, 64, 128]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(48, 50))
fig.suptitle('PPO Image Based Learning', fontsize=70, y=0.92)
# plt.tight_layout()
for update, ax in zip(updates, [ax1, ax2, ax3, ax4]):
    
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(labelsize=30)
    for agent in agents:
        path = '/home/alan/Documents/RL-Models/Paper/PPO_Line_{0}_{1}'.format(agent, update)
        print(path)
        dataset = load_results(path)
        x, y = ts2xy(dataset, xaxis='walltime_hrs')
        y = moving_average(y, 250)
        x = x[:-200]
        y = y[:-200]
        mean_rewards, mean_ts = [], []
        n_envs = path.split('/')[-1].split('_')[-2]
        n_steps = path.split('/')[-1].split('_')[-1]
        label = 'Agents: {0}'.format(n_envs)
        ax.plot(x, y, lw=5, label=label)

    ax.legend(fontsize=30, loc=0)
    ax.grid(linestyle="--")
    ax.set_title("Policy Update every {0} timesteps".format(n_steps), fontsize=50)
    ax.set_xlabel("Training Time [hrs]", fontsize=40)
    ax.set_ylabel("Mean Reward", fontsize=40)
    plt.sca(ax)
    plt.xticks(range(6))
    plt.ylim(0, 400)
