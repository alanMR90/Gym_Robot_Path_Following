#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:30:16 2020

@author: alan
"""


import numpy as np
import glob
import cv2
import bezier
from random import uniform, random, choice


def load_texture(size):
    """
    Parameters
    ----------
    size : 2D tuple
        World size, to adjust texture size.

    Returns
    -------
    texture : np.array
        Loaded texture array.
    """
    textures = glob.glob("/home/alan/Documents/Github/Gym_Robot_Path_Following/path_following/envs/textures/*.jpg",
                         recursive=True)
    texture = cv2.imread(choice(textures))
    texture = cv2.resize(texture, size, cv2.INTER_LINEAR)
    return texture


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def adjust_blur(image, k_size):
    image = cv2.blur(image, (k_size, k_size))
    return image


def generate_trajectory(n_points, size):
    """
    Parameters
    ----------
    n_points : int
        Number of points to discretize the bezier curve.

    Returns
    -------
    points : np.array
        Normalized array containing the generated random trajectory.
    """
    nodes = np.asfortranarray([
            [0.0, uniform(0.0, 0.4), uniform(0.2, 0.6), uniform(0.4, 0.8), uniform(0.6, 1.0), 1.0],
            [random(), random(), random(), random(), random(), random()],
                ])
    curve = bezier.Curve.from_nodes(nodes)
    s_vals = np.linspace(0.1, 0.9, n_points)
    points = curve.evaluate_multi(s_vals)
    points = points.transpose()
    points = size[0]*points.reshape((n_points, 2))
    orientations = calculate_orientations(s_vals, curve)
    trajectory = np.hstack((points, orientations))
    trajectory = np.unique(trajectory, axis=0)
    return trajectory


def calculate_orientations(s_vals, curve):
    orientations = []
    for i in range(len(s_vals)):
        hodo = curve.evaluate_hodograph(s_vals[i])
        tan = (hodo[1])/(hodo[0])
        angle = np.rad2deg(np.arctan(tan))
        normal = angle+90
        orientations.append(normal)
    return np.array(orientations)


def draw_trajectory(background, points, dr=False):
    """
    Parameters
    ----------
    background : np.array
        Background texture image.
    points : np.array
        Points to generate the trajectory.
    dr : Domain randomization flag, optional
        Enable to set random color and thickness of the drawed trajectory.
        The default is False.

    Returns
    -------
    world : np.array
        Array containing the background with the drawed trajectory.
    """
    points = points.astype(np.int32)
    if dr:
        traj_color = np.random.randint(low=64, high=255, size=3)
        goal_color = np.random.randint(low=64, high=255, size=3)
        traj_color = (int(traj_color[0]), int(traj_color[1]), int(traj_color[2]))
        goal_color = (int(goal_color[0]), int(goal_color[1]), int(goal_color[2]))
        thickness = np.random.randint(5, 10)

    else:
        traj_color = (255, 0, 0)
        goal_color = (0, 255, 0)
        thickness = 3
    world = cv2.polylines(background, [points], False, traj_color, thickness)
    world = cv2.circle(world, tuple(points[-1]), 12, goal_color, -1)
    return world


def render_world(n_points, size, dr):
    background = load_texture(size)
    trajectory = generate_trajectory(n_points, size)
    world = draw_trajectory(background, trajectory[:, 0:2], dr)
    if dr:
        world = adjust_gamma(world, np.random.uniform(0.4, 3.0))
        world = adjust_blur(world, np.random.randint(1, 5))
    return world, trajectory.astype(np.int32)


def render_laser_line(state, world, laser_length, laser_diam, window_shape):
    p1X = state[0] + int(0.5*laser_length*np.cos(np.deg2rad(state[2])))
    p1Y = state[1] + int(0.5*laser_length*np.sin(np.deg2rad(state[2])))
    p2X = state[0] + int(0.5*laser_length*np.cos(np.deg2rad(180+state[2])))
    p2Y = state[1] + int(0.5*laser_length*np.sin(np.deg2rad(180+state[2])))
    cv2.line(world, (p1X, p1Y), (p2X, p2Y), (0, 0, 255), laser_diam)

    y0 = state[1]-int(0.5*window_shape[0])
    y1 = state[1]+int(0.5*window_shape[0])
    x0 = state[0]-int(0.5*window_shape[1])
    x1 = state[0]+int(0.5*window_shape[1])

    y0 = np.clip(y0, 0, world.shape[0]-window_shape[0])
    x0 = np.clip(x0, 0, world.shape[1]-window_shape[1])
    y1 = np.clip(y1, window_shape[0], world.shape[0])
    x1 = np.clip(x1, window_shape[1], world.shape[1])

    observation = world[y0:y1, x0:x1]
    return world, observation


def render_laser_point(state, world, laser_diam, window_shape):
    cv2.circle(world, tuple(state), laser_diam, (0, 0, 255), -1)

    y0 = state[1]-int(0.5*window_shape[0])
    y1 = state[1]+int(0.5*window_shape[0])
    x0 = state[0]-int(0.5*window_shape[1])
    x1 = state[0]+int(0.5*window_shape[1])

    y0 = np.clip(y0, 0, world.shape[0]-window_shape[0])
    x0 = np.clip(x0, 0, world.shape[1]-window_shape[1])
    y1 = np.clip(y1, window_shape[0], world.shape[0])
    x1 = np.clip(x1, window_shape[1], world.shape[1])

    observation = world[y0:y1, x0:x1]
    return world, observation


def render_laser_traj(world, traj, laser_diam):
    traj = np.array(traj)
    traj = traj[:, 0:2].astype(np.int32)
    cv2.polylines(world, [traj], False, (0, 0, 0), laser_diam)
    return world


def calculate_reward_point(trajectory, tree, laser_traj, old_points):
    laser_traj = np.array(laser_traj)
    laser_traj = np.unique(laser_traj, axis=0)

    distances, _ = tree.query(laser_traj, distance_upper_bound=1e-5)
    points_in = (distances == 0).sum()

    if points_in > old_points:
        reward = 1
    else:
        reward = 0
    return reward, points_in


def calculate_reward_line(trajectory, tree, laser_traj, old_points):
    laser_traj = np.array(laser_traj)
    current_state = laser_traj[-1, :]
    laser_traj = np.unique(laser_traj, axis=0)
    laser_traj = laser_traj[:, 0:2]
    distances, index = tree.query(laser_traj, distance_upper_bound=1e-5)
    points_in = (distances == 0).sum()

    if points_in > old_points:
        reward = 1
        dist, idx = tree.query(current_state[0:2])
        ori_error = abs(trajectory[idx, 2] - current_state[2])
        r2 = np.float64(np.power(np.e, -0.12*ori_error))
        reward += r2
    else:
        reward = 0
    return reward, points_in
