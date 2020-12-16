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
from geomdl import BSpline, utilities
from geomdl.operations import normal
from numpy.random import randint


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


def generate_surface():
    surf = BSpline.Surface()

    control_points1 = [[0, 0, randint(0, 100)], [0, randint(0, 20), randint(0, 100)], [0, randint(50, 80), randint(0, 100)], [0, 100, randint(0, 100)],
                       [randint(20, 80), 0, randint(0, 100)], [randint(20, 80), randint(0, 20), randint(0, 100)], [randint(20, 80), randint(50, 80), randint(0, 100)], [randint(20, 80), 100, randint(0, 100)],
                       [100, 0, randint(0, 100)], [100, randint(0, 20), randint(0, 100)], [100, randint(20, 80), randint(0, 100)], [100, 100, randint(0, 100)]]

    # control_points2 = [[0, 0, 0], [0, 20, 0], [0, 70, 0], [0, 100, 0],
    #                    [50, 0, 30], [50, 30, 30], [50, 70, 30], [50, 100, 70],
    #                    [100, 0, 0], [100, 20, 0], [100, 70, 0], [100, 100, 0]]

    # control_points3 = [[0, 0, 0], [0, 20, 0], [0, 70, 0], [0, 100, 0],
    #                    [50, 0, 100], [50, 30, 30], [50, 70, 30], [50, 100, 80],
    #                    [100, 0, 0], [100, 20, 0], [100, 70, 0], [100, 100, 0]]

    # control_points4 = [[0, 0, 0], [0, 20, 0], [0, 70, 0], [0, 100, 0],
    #                    [50, 0, 0], [50, 30, 0], [50, 70, 0], [50, 100, 0],
    #                    [100, 0, 0], [100, 20, 0], [100, 70, 0], [100, 100, 0]]

    control_points5 = [[0, 0, 0], [0, 50, 0], [0, 50, 0], [0, 100, 0],
                       [50, 0, 0], [50, 50, 0], [50, 50, 0], [50, 100, 0],
                       [100, 0, 0], [100, 50, 0], [100, 50, 0], [100, 100, 0]]

    diam = np.random.randint(60, 100)
    r = diam/2
    x = 0.05*diam
    z = 4.*r/3.
    control_points6 = [[0, 0, 0], [0, 20, 0], [0, 70, 0], [0, 100, 0],
                       [x, 0, z], [x, 20, z], [x, 70, z], [x, 100, z],
                       [diam-x, 0, z], [diam-x, 20, z], [diam-x, 70, z], [diam-x, 100, z],
                       [diam, 0, 0], [diam, 20, 0], [diam, 70, 0], [diam, 100, 0]]

    # control_points_set = [control_points1, control_points2, control_points3, control_points4,
    #                       control_points5, control_points6]
    
    control_points_set = [control_points1]#, control_points5, control_points6]
    
    control_points = choice(control_points_set)

    if len(control_points) == 12:
        surf.degree_u = 2
        surf.degree_v = 3
        surf.set_ctrlpts(control_points, 3, 4)
    elif len(control_points) == 16:
        surf.degree_u = 3
        surf.degree_v = 3
        surf.set_ctrlpts(control_points, 4, 4)
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
    return surf


def get_norms(surf, timestep_limit):
    # create a 2D bezier curve
    curve = BSpline.Curve()
    curve.degree = 3

    ctrl_points = [[np.random.uniform(0, 0.2), np.random.uniform(0, 0.2)],
                   [np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)],
                   [np.random.uniform(0.25, 0.75), np.random.uniform(0.25, 0.75)],
                   [np.random.uniform(0.5, 0.75), np.random.uniform(0.5, 0.75)],
                   [np.random.uniform(0.8, 1), np.random.uniform(0.8, 1.0)]]

    curve.set_ctrlpts(ctrl_points)
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.sample_size = timestep_limit
    curve.evaluate()
    points_c = curve.evalpts
    norms = normal(surf, points_c)
    angles = []  # pitch, yaw
    traj = []  # x,y,z
    for i in range(len(norms)):
        nu = np.array(norms[i][1])
        yaw = 180.0*np.arctan2(np.abs(nu[1]), np.abs(nu[0]))/np.pi
        ca = np.linalg.norm(nu[0:2])
        pitch = 180.0*np.arctan2(nu[2], ca)/np.pi
        angles.append([pitch, yaw])
        point = np.array(norms[i][0])
        traj.append(np.array([point[0], point[1], point[2], pitch, yaw]))
    return traj, norms, curve
