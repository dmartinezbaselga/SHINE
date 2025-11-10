#!/usr/bin/env python3

import os, sys

import numpy as np

sys.path.append("..")
print(os.getcwd())

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import pandas as pd
import seaborn as sns
import ptitprince as pt

import imageio # animations
import pickle

from scripts.load_ros_data import *
from scripts.helpers import *
from scripts.metrics import add_num_obstacles, add_horizon_length
from scripts.compare import clean_method_name

# plt.rcParams['text.usetex'] = True
font = {
    'family' : 'serif',
    'serif' : ["Computer Modern Serif"],
    # 'weight' : 'bold',
    'size'   : 16}
plt.rc('font', **font)
fig_width = 7.02625 # /textwidth in latex in inches

cmap_viridis = matplotlib.cm.get_cmap('viridis')
cmap_viridis_r = matplotlib.cm.get_cmap('inferno')

COMPARE_VISUAL = True



def save_figure_as(fig, figure_folder, figure_name, ratio=1.0, verbose=True, save_pdf= True, save_png=True, **kwargs):
    fig.set_size_inches(fig_width, fig_width*ratio)

    os.makedirs(os.path.dirname(f'{figure_folder}{figure_name}'), exist_ok=True)

    if save_pdf:
        fig.savefig(f'{figure_folder}{figure_name}.pdf', bbox_inches='tight', **kwargs)

    if save_png:
        fig.savefig(f'{figure_folder}{figure_name}.png', bbox_inches='tight', **kwargs)

    if verbose:
        print(f'Saved figure as: {figure_folder}{figure_name}.pdf')


def get_planner_colors(cdf, palette="Set2"):
    planners = ["frenet-planner", "LMPCC~[4]", "GMPCCNO", "GMPCC", "Guidance-MPCC~[21]"]
    for p in range(len(planners)):
        planners[p] = clean_method_name(planners[p])
    colors = sns.color_palette(palette, n_colors=len(planners))
    return dict(zip(planners, colors))


def scatter(x, y, color, alpha_arr, **kwarg):
    r, g, b = matplotlib.colors.to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    plt.scatter(x, y, c=color, **kwarg)

"""
PLOTTING FUNCTIONALITY
"""
def subplot_signal(ax, signal, **kwargs):

    indices = range(0, len(signal))
    ax.step(indices, signal, **kwargs)

    ax.set_aspect('auto', adjustable='box')

def subplot_experiment_signal(ax, experiment_data, signal, **kwargs):

    d = merge_experiment_data(experiment_data, signal)
    indices = range(0, len(d))

    ax.step(indices, d, **kwargs)  # , marker=m[i % len(m)],markersize=15,markevery=5)

    ax.set_aspect('auto', adjustable='box')
    return ax

def compare_signals(experiment_data, signal_list, highlight_signal_list=[], filter=lambda value:False):
    fig = plt.figure()
    ax = plt.gca()
    m = ['.', 'x', '*']

    for i, signal in enumerate(signal_list):
        d = merge_experiment_data(experiment_data, signal)
        indices = range(0, len(d))

        # Filter if necessary
        indices = [index for index in indices if (not filter(d[index]))]
        d = [value for value in d if (not filter(value))]

        plt.step(indices, d, color='C0', linewidth=3, alpha=0.9)#, marker=m[i % len(m)],markersize=15,markevery=5)

    # HIGHLIGHTED
    for signal in highlight_signal_list:
        d = merge_experiment_data(experiment_data, signal)
        indices = range(0, len(d))

        # Filter if necessary
        indices = [index for index in indices if (not filter(d[index]))]
        d = [value for value in d if (not filter(value))]
        # plt.step(indices, d, color='C3', linestyle='--', linewidth=2)
        plt.step(indices, d, color='C3', linestyle='-', linewidth=2)

    ax.set_aspect('auto', adjustable='box')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')

    return fig
    # sns.boxplot(data=df) #y='{} (ms)'.format(signal_list[0])

def plot_multiple_signals(experiment_data, signal_list, x_name="x", y_name="y", use_subplots=False):
    if use_subplots:
        fig, axes = plt.subplots(nrows=len(signal_list), ncols=1)
    else:
        fig = plt.figure()
        ax = plt.gca()

    for i, signal in enumerate(signal_list):
        if use_subplots:
            ax = axes[i]
        d = merge_experiment_data(experiment_data, signal)
        indices = range(0, len(d))
        sns.lineplot(x=indices, y=d, palette="Set2", ax=ax)
    # plt.step(indices, d, linewidth=3, alpha=0.9)  # , marker=m[i % len(m)],markersize=15,markevery=5)

    # ax.set_aspect('auto', adjustable='box')

    return fig

def simulations_boxplot(simulation_data, data_name, simulation_name, simulations, do_strip_plot=True):
    return signal_boxplot(data_name, simulation_data[data_name], simulation_name, simulations, do_strip_plot)

# https://stackoverflow.com/questions/44552489/plotting-multiple-boxplots-in-seaborn
def signal_boxplot(data_name, data_list, range_name, range_list, do_strip_plot=True):
    fig = plt.figure()
    name_list = [str(name) for name in range_list]

    dfs = []
    for i, data_item in enumerate(data_list):
        dfs.append(pd.DataFrame({data_name:data_item, range_name:name_list[i]}))
    cdf = pd.concat(dfs)

    if do_strip_plot:
        pt.stripplot(x=data_name, y=range_name, data=cdf, ax=plt.gca(),
                     palette = "Set2", orient="h", jitter=1,zorder=0, dodge=True, width=0.9, color="C1")
    sns.boxplot(x=data_name, y=range_name, data=cdf, ax=plt.gca(),
            zorder=10, orient="h", width=.9, showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, showfliers=False,
            whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1)

    # pt.RainCloud(x=range_name, y=data_name, data=cdf,width_viol=0., width_box=0.9, ax=plt.gca(), palette = "Set2", orient="v", bw=0.2)

    plt.tight_layout()

    return fig

def comparative_boxplot_simulations_planners(simulation_data, data_name, simulations, planners):
    return comparitive_boxplot(data_name, simulation_data[data_name], "Pedestrians", simulations, "Planner", planners)

def comparitive_boxplot(x_name, x_list, y_name, y_list, category_name, category_list, do_strip_plot=True):
    print("Comparative boxplot of " + x_name + " vs " + y_name + " comparing for each the " + category_name)
    name_list = [str(name) for name in category_list]

    n_simulations = 1
    prev_value = category_list[0]
    for value in category_list[1:]:
        if value == prev_value:
            n_simulations += 1
        else:
            break
    n_planners = int(len(x_list) / n_simulations)

    dfs = []
    for i, data_item in enumerate(x_list):
        dfs.append(pd.DataFrame({x_name:data_item, category_name:name_list[i], y_name:y_list[i]}))
    cdf = pd.concat(dfs)

    print(category_name)
    fig = plt.figure()
    colors = get_planner_colors(cdf)
    stripplot = None
    width = 0.7
    if do_strip_plot:
        stripplot = pt.stripplot(x=x_name, y=y_name, hue=category_name, data=cdf, ax=plt.gca(),
                     palette=colors, orient="h", jitter=1,zorder=0, dodge=True, width=width)
    boxplot = sns.boxplot(x=x_name, y=y_name, hue=category_name, data=cdf, ax=plt.gca(),
                     zorder=10, orient="h", width=width, showcaps=True, boxprops={'facecolor':'none', "zorder":10},\
                     showfliers=False, whiskerprops={'linewidth':2, "zorder":10}, saturation=1)

    handles, labels = stripplot.get_legend_handles_labels()
    # Remove the boxplot handles and labels from the legend
    handles = handles[n_planners:]
    labels = labels[n_planners:]
    plt.legend(handles, labels, ncol=1#math.ceil(n_planners/4),
                , loc='upper right',
               borderpad=0.3, labelspacing=0.1, handlelength=0.3, handletextpad=0.7, borderaxespad=0.3)

    sns.despine(offset=3)

    plt.tight_layout()

    return fig

def animate_trajectories(experiment_data, experiment_nr, simulation, planner_name, subfolder=""):
    _, _, figure_folder = get_folder_names()
    save_folder = figure_folder + "../analysis/" + simulation + "-" + planner_name + "/" + subfolder

    experiment = experiment_data[experiment_nr]
    temp = dict()
    add_num_obstacles([experiment], temp)
    num_obstacles = temp["num obstacles"]

    N = len(experiment["vehicle_pose"][:, 0])
    DT = 5
    speed_up = 3
    repeat_at_collision = 10

    # Determine the limits
    x_min = 1e9
    y_min = -3
    x_max = -1e9
    y_max = 3
    x_min, x_max = find_min_max(x_min, x_max, experiment["vehicle_pose"][:, 0])
    y_min, y_max = find_min_max(y_min, y_max, experiment["vehicle_pose"][:, 1])

    for m in range(num_obstacles):
        try:
            x_min, x_max = find_min_max(x_min, x_max, experiment[f"obstacle_{m}_pose"][:, 0])
            y_min, y_max = find_min_max(y_min, y_max, experiment[f"obstacle_{m}_pose"][:, 1])
        except KeyError:
            pass
        except ValueError:
            pass

    frame_id = 0
    repeat_frame = 0
    for t in range(0, N, DT):
        fig = plt.figure(figsize=(x_max - x_min, y_max - y_min), tight_layout=True)
        ax = plt.gca()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        collision_case = False
        if t > 0:
            # Is there a collision in this time frame?
            for t_range in range(t-DT, t):
                collision_case = collision_case or experiment["max_intrusion"][t_range] > 0.05

        # Plot the road constraints
        ax.plot([x_min, x_max], [3., 3.], linestyle='--', color='k', linewidth=2.0)
        ax.plot([x_min, x_max], [-3., -3.], linestyle='--', color='k', linewidth=2.0)

        if collision_case:
            plot_trajectory(ax, experiment["vehicle_pose"], t_start=t, t_final=t, color='r', linewidth=3.0)
        else:
            plot_trajectory(ax, experiment["vehicle_pose"], t_start=t, t_final=t, color='C0', linewidth=3.0)

        for m in range(num_obstacles):
            try:
                plot_trajectory(ax, experiment[f"obstacle_{m}_pose"], t_start=t, t_final=t, color="C2", linewidth=1.5)
            except KeyError:
                pass
            except ValueError:
                pass
        save_figure_as(fig, save_folder, str(frame_id), ratio=(y_max - y_min)/(x_max - x_min), verbose=False, save_pdf=False)
        frame_id += 1

        if collision_case:
            for repeat_i in range(repeat_at_collision):
                save_figure_as(fig, save_folder, str(frame_id + repeat_i), ratio=(y_max - y_min) / (x_max - x_min), verbose=False, save_pdf=False)

            frame_id += repeat_at_collision

    with imageio.get_writer(save_folder + f"{experiment_nr}.gif", mode='I', duration=1 / 20. * DT / speed_up) as writer:
        for frame in range(frame_id):
            image = imageio.imread(save_folder + f"{frame}.png")
            writer.append_data(image)

    for frame in range(frame_id):
        os.remove(save_folder + f"{frame}.png")

def plot_trajectory(ax, pos_vector, t_start=0, t_final=-1, show_trace=True, **kwargs):
    if t_final == -1:
        t_final = len(pos_vector[:, 0])

    ax.scatter(pos_vector[t_start:t_final+1, 0], pos_vector[t_start:t_final+1, 1], alpha=1., s=200, marker='.', label='_nolegend_', **kwargs)

    if show_trace:
        ax.plot(pos_vector[0:t_final+1, 0], pos_vector[0:t_final+1, 1], alpha=0.3, linestyle='--', **kwargs)#color='C0', linewidth=3.0, alpha=1.0)

def plot_experimental_trajectories(ax, pos_vector, t_start=0, t_final=-1, show_trace=True, radius=0.325, text='', **kwargs):
    if t_final == -1:
        t_final = len(pos_vector[:, 0])

    t = range(t_start,t_final + 1)
    normalized_time = (t - np.min(t)) / (np.max(t) - np.min(t))

    for i in range(t_start, t_final):
        circle = plt.Circle((-pos_vector[i, 1], pos_vector[i, 0]), radius=radius, alpha=0.001 + 0.05 * (1 - normalized_time[i - t_start]), **kwargs)
        ax.add_patch(circle)

    if text != '':
        plt.text(-pos_vector[t_start, 1] + 0.5, pos_vector[t_start, 0]-0.15, text, fontsize=26, weight='bold', **kwargs)

    if show_trace:
        ax.plot(-pos_vector[t_start, 1], pos_vector[t_start, 0], marker='o', alpha=0.7, color='k', markersize=10, label='_nolegend_')
        ax.plot(-pos_vector[t_start:t_final + 1, 1], pos_vector[t_start:t_final + 1, 0], marker='.', markersize=4, markevery=10, alpha=1.,
                label='_nolegend_', **kwargs)


def plot_plan(data, t, plan_list, highlighted_plan_list=[]):

    fig = plt.figure(tight_layout=True)
    ax = plt.gca()

    # Choose which plan to plot
    experiment_data = data[0]
    time_instance = t

    plan_name = lambda name, k: name + str(k)

    for index, current_plan_name in enumerate(plan_list):
        # Plot the plan along the horizon
        horizon = 0
        for k in range(0, 1000):
            if not plan_name(current_plan_name, k) in experiment_data:
                horizon = k
                break

        plan = np.zeros((2, horizon))
        for k in range(0, horizon):
            plan[:, k] = experiment_data[plan_name(current_plan_name, k)][time_instance, :]

        plt.plot(plan[0, :], plan[1, :], color='C0', linewidth=3.0, alpha=1.0)
        # plt.scatter(plan[0, :], plan[1, :], color='C0', s=1200, alpha=0.4, label='_nolegend_')
        plt.scatter(plan[0, :], plan[1, :], color='C0', s=200, alpha=1.0, marker='.', label='_nolegend_')

    for index, current_plan_name in enumerate(highlighted_plan_list):
        # Plot the plan along the horizon
        horizon = 0
        for k in range(0, 1000):
            if not plan_name(current_plan_name, k) in experiment_data:
                horizon = k
                break

        plan = np.zeros((2, horizon))
        for k in range(0, horizon):
            plan[:, k] = experiment_data[plan_name(current_plan_name, k)][time_instance, :]

        color_index = index + len(plan_list)
        plt.plot(plan[0, :], plan[1, :], linestyle='--', color='C' + str(color_index), alpha=1.0, linewidth=3.0)
        plt.scatter(plan[0, :], plan[1, :], color='C' + str(color_index), s=100, alpha=1.0, linewidths=8.0, marker='x')

    # Plot the obstacle position
    add_num_obstacles(data, experiment_data)
    for obstacle in range(experiment_data["num obstacles"]):
        pose_name = f"disc_{obstacle}_pose"
        plt.plot(experiment_data[pose_name][time_instance, 0], experiment_data[pose_name][time_instance, 1],
                 marker='.', markersize=15, linewidth=3.0,
                 color='k')
        ax.annotate("Obstacle", xy=(experiment_data[pose_name][time_instance, 0]+0.25, experiment_data[pose_name][time_instance, 1]),
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16
                    )

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    return fig

def plot_linearized_constraint_regions(data, t, plan_list, color_start=0):
    subfigure_mode = True
    if subfigure_mode:
        fig, axs = plt.subplots(2, tight_layout=True)
    else:
        fig = plt.figure(tight_layout=True)
        ax = plt.gca()
        axs = [ax]

    experiment_data = data[0]
    add_horizon_length(data, experiment_data)

    beta = 0.25
    r_obs = 0.5
    r = 0.325 + r_obs

    for k in range(experiment_data["horizon length"]):

        for index, plan in enumerate(plan_list):
            if subfigure_mode:
                ax = axs[index]

            if k == 0 and index == 0:
                plan_positions = np.zeros((2, 2, experiment_data["horizon length"]))

            pos = experiment_data[plan + str(k)][t, :]
            obs = experiment_data["disc_0_pose"][t + k*4, :] # one obstacle
            plan_positions[index, :, k] = pos
            color_idx = (index + color_start) * 2

            # Plot a polygon
            coords, points_on_line = get_points_in_polygon(pos, obs, beta, r_obs, r)
            if len(coords) == 0:
                continue
            p = Polygon(coords, facecolor='C' + str(color_idx), alpha=0.1, zorder=0)
            ax.add_patch(p)

            # Plot the constraint
            ax.plot([points_on_line[0][1], points_on_line[1][1]], [points_on_line[0][0], points_on_line[1][0]],
                     color='C' + str(color_idx), linewidth=1.0, alpha=0.6, linestyle='--', label='_nolegend_')

    for index, plan in enumerate(plan_list):
        if subfigure_mode:
            ax = axs[index]
        color_idx = (index + color_start) * 2

        ax.plot(plan_positions[index, 0, :], plan_positions[index, 1, :], color='C' + str(color_idx),
                 linewidth=3.0, alpha=1.0, zorder=0.1)
        ax.scatter(plan_positions[index, 0, :], plan_positions[index, 1, :], color='C' + str(color_idx),
                    marker='.', s=200, alpha=1.0, label='_nolegend_', zorder=0.1)


    for i, ax in enumerate(axs):
        for k in range(experiment_data["horizon length"]):
            ax.plot(experiment_data["disc_0_pose"][t + k*4, 0], experiment_data["disc_0_pose"][t + k*4, 1],
                 marker='.', markersize=15, linewidth=3.0, alpha=1-k/experiment_data["horizon length"],
                 color='k')
        ax.annotate("Obstacle",
                    xy=(experiment_data["disc_0_pose"][t, 0]+0.2, experiment_data["disc_0_pose"][t, 1]-0.025),
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16
                    )
        ax.set_aspect('equal', adjustable='box')
        if i == len(axs) - 1:
            ax.set_xlabel('X (m)')
        else:
            ax.set_xticks([])
        ax.set_ylabel('Y (m)')

    return fig

def runtime_boxplot(data, save_path):
    # Plot
    fig_comp_time_box, ax_comp_time_box = plt.subplots()
    fig_comp_time_box.tight_layout(rect=[0, 0.03, 1, 0.95])
    runtime_data = [float(i) for i in data["runtime_control_loop"] if i != -1.]
    ax_comp_time_box.boxplot(runtime_data, showfliers=True, whis=[5, 95])

    # Set & save
    fig_comp_time_box.suptitle('Computation times boxplot')
    ax_comp_time_box.set_title('')
    ax_comp_time_box.set_xlabel('Number of Obstacles')
    ax_comp_time_box.set_ylabel('Time [ms]')
    ax_comp_time_box.grid(color='lightgray', linestyle='-', linewidth=0.1)
    fig_comp_time_box.savefig(f'{save_path}/runtime_boxplot.svg', bbox_inches='tight', format="svg")

def runtime_visual(experiment_data, metrics, save_path):
    runtime_data = merge_experiment_data(experiment_data, "runtime_control_loop")
    df = pd.DataFrame(runtime_data)
    sns.set_style("whitegrid")

    sns.boxplot(y='runtime (ms)', data=df)


def trajectory_visual(experiment_data, save_path, color_index=0):

    min_x = 1e3
    max_x = -1e3
    min_y = 1e3
    max_y = -1e3

    for experiment in experiment_data:
        max_x = max(max_x, np.max(experiment["vehicle_pose"][:, 0]))
        min_x = min(min_x, np.min(experiment["vehicle_pose"][:, 0]))
        max_y = max(max_y, np.max(experiment["vehicle_pose"][:, 1]))
        min_y = min(min_y, np.min(experiment["vehicle_pose"][:, 1]))

        fig = plt.scatter(experiment["vehicle_pose"][:, 0], experiment["vehicle_pose"][:, 1],
                    c=color_index*np.ones((len(experiment["vehicle_pose"]), )),
                    vmin=0,
                    vmax=5,
                    alpha=0.15,
                    cmap='viridis'
                    )

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    min_x = 0
    max_x = 30
    min_y = -3
    max_y = 3

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    # print(f"{min_x}, {max_x} | {min_y}, {max_y}")
    return fig


def visualize_simulation_results(simulation):
    metric_folder = os.getcwd() + "/metrics/" + simulation

    print(bcolors.OKGREEN + "Loading Planners..." + bcolors.ENDC)
    data = []
    metrics = []
    save_paths = []
    for i, filename in enumerate(os.scandir(metric_folder)):
        if filename.is_file():
            if filename.name.endswith(".json"):
                print_value("Planner", str(filename), tab=True)
                planner_name = filename.name.split(".")[0]

                experiment_data, metrics_single, save_path = load_simulation_and_planner(simulation, planner_name)
                save_path += simulation
                os.makedirs(save_path, exist_ok=True)

                data.append(experiment_data)
                metrics.append(metrics_single)
                save_paths.append(save_path)

        if i == 0:
            break

    # Plots per planner
    runtimes = []
    method_names = []
    for i in range(len(data)):
        runtimes.append(merge_experiment_data(data[i], "runtime_control_loop"))

        method_names.append(metrics[i]["name"])
        # metrics[i]["name"]

    plot_plan(experiment_data, ["solver99_plan", "solver0_plan", "solver1_plan", "vehicle_plan_"])
    # plot_plan(experiment_data, ["solver0_plan", "solver1_plan", "solver99_plan"])
    # plot_plan(experiment_data, ["solver0_plan", "solver1_plan", "solver2_plan", "solver3_plan", "solver99_plan"])

    # compare_signals(data, ["best_planner_idx"], save_path, "planner_idx")
    # compare_signals(data, ["objective_1", "objective_2", "objective_3", "objective_99", "objective"], save_path, "objectives")
    # compare_signals(data, ["objective_1", "objective_99", "objective"], save_path, "objectives")

    # df = pd.DataFrame({'runtimes': runtimes, 'method_names': method_names})

    sns.set_style("whitegrid")
    # sns.boxplot(x="method_names", y="runtimes", data=df)#, data=runtimes)

    # fig, axs = plt.subplots(len(data), 1)
    # for i in range(len(data)):
    #     plt.sca(axs[i])
    #     trajectory_fig = trajectory_visual(data[i], save_paths[i], i)
    #     plt.title(metrics[i]["name"])
    #
    # if trajectory_fig is not None:
    #     plt.savefig(f'{save_paths[0]}/trajectory_comparison.png', bbox_inches='tight', format="png")




def main(simulation, planner_name):

    visualize_simulation_results(simulation)

    # runtime_boxplot(data, save_path)
    # if COMPARE_VISUAL:
    #     compare_trajectory_visual(simulation)
    # else:
    #     trajectory_visual(experiment_data, save_path)


    plt.show()

if __name__ == '__main__':
    simulation = sys.argv[1]
    planner_name = sys.argv[2]
    main(simulation, planner_name)