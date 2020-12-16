#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:43:44 2020

@author: alan
"""
import os
import numpy as np

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from colorama import Fore, Style


class SaveOnBestTrainingReward(BaseCallback):
    
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingReward, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_')
        self.best_mean_reward = -np.inf
        self.model_index = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            # os.makedirs(self.save_path, exist_ok=True)
            pass

    def _on_training_start(self) -> None:
        print("Saving Initial model to {}".format(os.path.join(self.log_dir, "initial_model.zip")))
        self.model.save(os.path.join(self.log_dir, "initial_model.zip"))

    def _on_training_end(self) -> None:
        print("Saving End model to {}".format(os.path.join(self.log_dir, "end_model.zip")))
        self.model.save(os.path.join(self.log_dir, "end_model.zip"))


    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  self.model_index += 1
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print(Fore.YELLOW + "Saving new best model to {}".format(self.save_path+str(self.model_index)) + Style.RESET_ALL)
                  self.model.save(self.save_path+str(self.model_index)+".zip")
        return True
