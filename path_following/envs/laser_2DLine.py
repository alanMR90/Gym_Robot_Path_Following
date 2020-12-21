#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:45:06 2020

@author: alan
"""


import numpy as np
import gym
import cv2
from gym import spaces
from path_following.envs.envUtils import utils
from gym.envs.classic_control import rendering
from scipy.spatial import cKDTree
from colorama import Fore, Style


class Laser2DLine(gym.Env):
    """
    Custom environment that follows gym interface. This env renders a 2D
    texture and plots a laser. The agent must learn how to follow a trajectory
    with the laser keep a orientation normal to the trajectory.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(Laser2DLine, self).__init__()
        "Initialize environment variables"
        params = {"window_shape": (60, 60, 3),
                  "world_size": (600, 600),
                  "laser_length": 40,
                  "laser_diam": 5,
                  "domain_randomization": True,
                  "n_points": 600,
                  "timesteps": 1000
                  }

        self.window_shape = params["window_shape"]
        self.world_size = params["world_size"]
        self.laser_length = params["laser_length"]
        self.laser_diam = params["laser_diam"]
        self.dr = params["domain_randomization"]
        self.n_points = params["n_points"]
        self.timestep_limit = params["timesteps"]
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.window_shape,
                                            dtype=np.uint8)
        self.viewer = None

    def reset(self):
        self.world, self.trajectory = utils.render_world(self.n_points,
                                                         self.world_size,
                                                         self.dr)
        if self.dr:
            self.laser_length = np.random.randint(10, self.window_shape[0])
            self.laser_diam = np.random.randint(5, 10)

        self.KDT = cKDTree(self.trajectory[:, 0:2])
        self.state = self.trajectory[0, :]
        self.goal = self.trajectory[-1, :]
        self.timestep = 0
        self.c_reward = 0
        self.points_in = 0
        self.laser_traj = []
        self.laser_traj.append(self.state.astype(np.int32))
        self.worldL, self.observation = utils.render_laser_line(self.state,
                                                                np.copy(self.world),
                                                                self.laser_length,
                                                                self.laser_diam,
                                                                self.window_shape)
        return self.observation

    def step(self, action):
        self.timestep += 1
        info = {}
        info["status"] = "ok"
        done = False
        reward = 0
        self.state += action - np.ones(action.shape, dtype=np.int32)
        self.state[0:2] = np.clip(self.state[0:2], [0, 0], [self.world_size])
        if self.state[2] == 360:
            self.state[2] = 0
        if self.state[2] == -1:
            self.state[2] = 359
        self.laser_traj.append(self.state.astype(np.int32))
        self.worldL = utils.render_laser_traj(np.copy(self.world),
                                              self.laser_traj,
                                              self.laser_diam)
        self.worldL, self.observation = utils.render_laser_line(self.state,
                                                                self.worldL,
                                                                self.laser_length,
                                                                self.laser_diam,
                                                                self.window_shape)

        reward, self.points_in, pos_error, ori_error = utils.calculate_reward_line(self.trajectory, self.KDT,
                                                             self.laser_traj, self.points_in)
        dist2Goal = np.linalg.norm(self.goal[0:2]-self.state[0:2])

        if dist2Goal < 20:
            done = True
            info["status"] = "goal reached"
            reward += 100
        elif self.timestep > self.timestep_limit:
            info["status"] = "timesteps limit reached"
            done = True
        self.c_reward += reward
        info["state"] = self.state
        info["world"] = self.worldL
        info["timesteps"] = self.timestep
        info["cReward"] = self.c_reward
        info["pos_error"] = pos_error
        info["ori_error"] = ori_error

        if done and info["status"] == "goal reached":
            print(Fore.GREEN + "reward: {0:.2f}, timesteps: {1}, status: {2}\
                  ".format(info["cReward"], info["timesteps"], info["status"]) + Style.RESET_ALL)
        return self.observation, reward, done, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        if mode == "human":
            self.viewer.imshow(self.observation)

        elif mode == "rgb_array":
            render_frame = np.copy(self.worldL)
            render_frame[5:self.window_shape[1]+5, 5:self.window_shape[0]+5] = self.observation
            cv2.rectangle(render_frame, (0,0), (self.window_shape[1]+10, self.window_shape[0]+10),
                         (0,0,0), 5)
            return render_frame
            #return self.worldL, self.observation

    def close(self):
        pass
