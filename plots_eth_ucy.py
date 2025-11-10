import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.animation as animation
import os
import struct
from simulator import Position, read_scenario
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def save_scenario2D(obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj):
    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    fontsize = 13.
    radius=0.325
    MIN_X = -3.
    MAX_X = 3.
    MIN_Y = -10.
    MAX_Y = 10.
    # Obstacles previous
    for i_obs in range(len(obstacles_previous)):
        for i in range(len(obstacles_previous[i_obs])):
            circle = plt.Circle((-obstacles_previous[i_obs][i].y, obstacles_previous[i_obs][i].x), radius=radius, alpha=(0.05 + 0.45 * (1 - (i/len(obstacles_previous[i_obs]))/2.)), color="C2")
            ax.add_patch(circle)
        plt.plot([-o.y for o in obstacles_previous[i_obs]], [o.x for o in obstacles_previous[i_obs]], "-", color="C2")
        plt.text(-obstacles_previous[i_obs][0].y + 0.5, obstacles_previous[i_obs][0].x-0.15, "O" + str(i_obs), fontsize=fontsize, weight='bold', color="C2")
    # Obstacles future
    for i_obs in range(len(obstacles_vec)):
        for i in range(1,len(obstacles_vec[i_obs])):
            circle = plt.Circle((-obstacles_vec[i_obs][i].y, obstacles_vec[i_obs][i].x), radius=radius, alpha=(0.05 + 0.45 * (0.5 - (i/len(obstacles_vec[i_obs]))/2.)), color="C2")
            ax.add_patch(circle)
        plt.plot([-o.y for o in obstacles_vec[i_obs]], [o.x for o in obstacles_vec[i_obs]], "-", color="C2")
    # Previous traj
    for i in range(len(human_previous)):
        circle = plt.Circle((-human_previous[i].y, human_previous[i].x), radius=radius, alpha=(0.05 + 0.45 * (1 - (i/len(human_previous))/2.)), color="C1")
        ax.add_patch(circle)
    plt.plot([-o.y for o in human_previous], [o.x for o in human_previous], "-", color="C1")
    plt.text(-human_previous[0].y + 0.5, human_previous[0].x-0.15, "Robot", fontsize=fontsize, weight='bold', color="C1")
    # Ours
    for i in range(1,len(ours_traj)):
        circle = plt.Circle((-ours_traj[i].y, ours_traj[i].x), radius=radius, alpha=(0.05 + 0.45 * (0.5 - (i/len(ours_traj))/2.)), color="C3")
        ax.add_patch(circle)
    plt.plot([-o.y for o in ours_traj], [o.x for o in ours_traj], "-", color="C3")
    plt.text(-ours_traj[0].y + 0.5, ours_traj[0].x-0.15, "Ours", fontsize=fontsize, weight='bold', color="C3")
    # Baseline
    for i in range(1,len(baseline_traj)):
        circle = plt.Circle((-baseline_traj[i].y, baseline_traj[i].x), radius=radius, alpha=(0.05 + 0.45 * (0.5 - (i/len(baseline_traj))/2.)), color="C4")
        ax.add_patch(circle)
    plt.plot([-o.y for o in baseline_traj], [o.x for o in baseline_traj], "-", color="C4")
    plt.text(-baseline_traj[0].y + 0.5, baseline_traj[0].x-0.15, "Baseline", fontsize=fontsize, weight='bold', color="C4")


    plt.xlim([MIN_X, MAX_X])
    plt.ylim([MIN_Y, MAX_Y])
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    plt.axis('off')
    fig.tight_layout()
    ax.set_aspect('equal')
    plt.show()

def plot_cylinder(ax, center_x, center_y, z_center, radius, alpha, color):
    z = np.linspace(z_center-0.2, z_center+0.2, 10)
    theta = np.linspace(0, 2*np.pi, 10)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)

def save_scenario3D(obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj, scenario_i, batch_i):
    print(scenario_i, batch_i)
    plt.close()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    t_prev = np.linspace(0, 7, 8)*0.4
    t_fut = np.linspace(7, 20, 14)*0.4
    fontsize = 13.
    radius=0.325
    MIN_X = -3.
    MAX_X = 3.
    MIN_Y = -10.
    MAX_Y = 10.
    remove_obs = []
    if scenario_i == 2 and batch_i == 0:
        remove_obs = [3,7,2,1,5]
    elif scenario_i == 26 and batch_i == 0:
        remove_obs = [3,7,2,1,5]
    # Obstacles previous
    for i_obs in range(len(obstacles_previous)):
        if not i_obs in remove_obs:
            for i in range(len(obstacles_previous[i_obs])):
                plot_cylinder(ax, -obstacles_previous[i_obs][i].y, obstacles_previous[i_obs][i].x, t_prev[i], radius=radius, alpha=(0.05 + 0.6 * (1 - (i/len(obstacles_previous[i_obs]))/2.)), color="C2")
            ax.text(-obstacles_previous[i_obs][0].y + 0.5, obstacles_previous[i_obs][0].x-0.15, 0, "O" + str(i_obs))
        # plt.text(-obstacles_previous[i_obs][0].y + 0.5, obstacles_previous[i_obs][0].x-0.15, "O" + str(i_obs), fontsize=fontsize, weight='bold', color="C2")
    # Obstacles future
    for i_obs in range(len(obstacles_vec)):
        if not i_obs in remove_obs:
            for i in range(1,len(obstacles_vec[i_obs])):
                plot_cylinder(ax, -obstacles_vec[i_obs][i].y, obstacles_vec[i_obs][i].x, t_fut[i], radius=radius, alpha=(0.05 + 0.6 * (0.5 - (i/len(obstacles_vec[i_obs]))/2.)), color="C2")
    # Previous traj
    for i in range(len(human_previous)):
        plot_cylinder(ax, -human_previous[i].y, human_previous[i].x, t_prev[i], radius=radius, alpha=(0.05 + 0.6 * (1 - (i/len(human_previous))/2.)), color="C1")
    # plt.text(-human_previous[0].y + 0.5, human_previous[0].x-0.15, "Robot", fontsize=fontsize, weight='bold', color="C1")
    # Ours
    for i in range(1,len(ours_traj)):
        plot_cylinder(ax, -ours_traj[i].y, ours_traj[i].x, t_fut[i], radius=radius, alpha=(0.05 + 0.6 * (0.5 - (i/len(ours_traj))/2.)), color="C3")
    # plt.text(-ours_traj[0].y + 0.5, ours_traj[0].x-0.15, "Ours", fontsize=fontsize, weight='bold', color="C3")
    # Baseline
    for i in range(1,len(baseline_traj)):
        plot_cylinder(ax, -baseline_traj[i].y, baseline_traj[i].x, t_fut[i], radius=radius, alpha=(0.05 + 0.6 * (0.5 - (i/len(baseline_traj))/2.)), color="C4")
    # plt.text(-baseline_traj[0].y + 0.5, baseline_traj[0].x-0.15, "Baseline", fontsize=fontsize, weight='bold', color="C4")


    # plt.xlim([MIN_X, MAX_X])
    # plt.ylim([MIN_Y, MAX_Y])
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    plt.axis('off')
    fig.tight_layout()
    plt.show()


mode = "3D"

for batch_id in range(5):
    for scenario_i in range(128):
        scenario_info = read_scenario(batch_id, scenario_i, "good_scenarios")
        while scenario_info is None:
            scenario_i = scenario_i+1
            if scenario_i > 127:
                batch_id = batch_id+1
                scenario_i = -1
            if batch_id == 5:
                exit(0)
            scenario_info = read_scenario(batch_id, scenario_i, "good_scenarios")
        obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj = scenario_info
        if mode == "2D":
            save_scenario2D(obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj)
        elif mode == "3D":
            save_scenario3D(obstacles_previous, obstacles_vec, goals, human_truth, human_previous, baseline_traj, ours_traj, scenario_i, batch_id)
