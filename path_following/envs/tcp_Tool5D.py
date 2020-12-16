#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:45:06 2020

@author: alan
"""


import numpy as np
import gym
from gym import spaces
from path_following.envs.envUtils import utils
from colorama import Fore, Style


class Tool5D(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(Tool5D, self).__init__()
        "Initialize environment variables"
        self.tlp = 100
        self.blp = -100
        self.tla = 180
        self.bla = -180
        self.timestepLimit = 1000
        self.test = False

        low_bound = np.array([self.blp, self.blp, self.blp, self.bla, self.bla])
        high_bound = np.array([self.tlp, self.tlp, self.tlp, self.tla, self.tla])

        low_bound = low_bound.reshape((5, 1))
        high_bound = high_bound.reshape((5, 1))
        self.observation_space = spaces.Box(low_bound, high_bound, dtype=np.float64)

        # Action space tuple of 3 actions for each DoF
        self.action_space = spaces.Box(-5.0, +5.0, (5,), dtype=np.float32)
        self.episode = 0
        self.viewer = None

    def reset(self):
        self.episode = 0
        self.timestep = 0
        self.cRew = 0.0
        self.surf = utils.generate_surface()
        self.traj, self.norms, self.curve = utils.get_norms(self.surf, self.timestepLimit)
        if self.test:
            offset = np.zeros((5,))
        else:
            offset = np.random.uniform(self.blp/2, self.tlp/2, (5,))
        self.pos = np.array(self.traj[self.timestep]) + offset
        self.pos = np.reshape(self.pos, (5, 1))
        self.goal = np.array(self.traj[self.timestep])
        self.goal = np.reshape(self.goal, (5, 1))
        self.state = self.goal - self.pos
        return self.state

    def step(self, action):
        action = action.reshape((5, 1))
        self.pos += action
        info = {"status": "ok"}
        reward = 0.0
        done = False
        self.state = self.goal - self.pos
        norm = np.linalg.norm(self.state)
        r = np.float64(np.power(np.e, -0.1*norm))
        if self.timestep > self.timestepLimit:
            done = True
            self.episode += 1
            info["status"] = "Timestep limit"
#        elif norm < 0.5:
#            done = True
#            info["status"] = "reached minimum norm"
        reward += r
        self.cRew += reward
        self.timestep += 1
        if self.timestep > self.timestepLimit-1:
            self.goal = np.array(self.traj[self.timestepLimit-1])
        else:
            self.goal = np.array(self.traj[self.timestep])
        self.goal = np.reshape(self.goal, (5, 1))

        info["pos"] = self.pos
        info["state"] = self.state
        info["cRew"] = self.cRew
        info["step"] = self.timestep
        info["episode"] = self.episode

        if done and norm < 5:
            info['status'] = 'goal reached'
            print(Fore.GREEN + "reward: {0:.2f}, timesteps: {1}, status: {2}\
                  ".format(info["cRew"], info["step"], info["status"]) + Style.RESET_ALL)
        return self.state, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
