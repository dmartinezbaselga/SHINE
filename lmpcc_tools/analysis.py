# This file is intended for manual analysis of experiments
import random
import sys, os
sys.path.append("..")

import numpy as np

import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from scripts.load_ros_data import *
import scripts.visuals, scripts.metrics
from scripts.helpers import *
from scripts.compare import clean_simulation_name, clean_method_name

# @todo: Put somewhere


"""
Analysis functions (select through argument)
"""


# I had an issue where the obstacle data is off every 2 iterations
def fix_jumps_in_data(data):
    add = "disc_0_pose" not in data.keys()

    for i in range(100):
        if f"disc_{i+add}_pose" not in data.keys():
            continue

        values_too_large = sum([1 for val in data[f"disc_{i+add}_pose"] if val[0] > 100.])
        if values_too_large >= len(data[f"disc_{i+add}_pose"]) / 2. - 1:
            data[f"disc_{i}_pose"] = data[f"disc_{i+add}_pose"][0::2]

def load_data(simulation, planner_name, fix_jumps=False):
    filename = simulation + "_" + planner_name
    data_folder, metric_folder, figure_folder = get_folder_names()
    data_path = data_folder + filename + ".txt"

    metric_folder = metric_folder + simulation
    metric_path = metric_folder + "/" + planner_name + ".json"
    with open(metric_path) as f:
        metrics = dict(json.load(f))

    data = load_ros_data(data_path)
    if fix_jumps:
        fix_jumps_in_data(data)

    settings = get_settings()
    experiment_data = split_ros_data(data, num_experiments=235, filter_faulty=settings.filter_faulty)
    return data, experiment_data, metrics, figure_folder

def boxplot_of_multiple_simulations_and_planners(simulations, planners, figure_base_name, reload_cache=False, xmin=11.5, xmax=25.):
    data_folder, metric_folder, figure_folder = get_folder_names()

    simulations *= len(planners)  # Repeat the simulations for each planner

    def duplicate(testList, n):
        return [ele for ele in testList for _ in range(n)]

    n_planners = len(planners)
    planners = duplicate(planners, int(len(simulations) / len(planners)))  # Duplicate planners repeating the same planner N times

    assert len(planners) == len(simulations), "after copying there should be equal number of entries in simulations and planners"

    data_names = ["Task Duration", "Infeasibility"]
    data_functions = [
        lambda experiment_data, metrics: [e for e in merge_experiment_data(experiment_data, "metric_duration") if (e > 10. and e < 39.)],
        lambda experiment_data, metrics: metrics['num infeasible']]

    # Load the data from a cache if it exists (this prevents the slow loading of all planner data)
    if not reload_cache:
        simulation_data, cache_exists = load_from_cache(simulations, planners)
    else:
        cache_exists = False

    if not cache_exists:
        simulation_data = merge_simulation_data(simulations, planners, data_names, data_functions)
        save_in_cache(simulations, planners, simulation_data)

    for sim in range(len(simulations)):
        simulations[sim] = clean_simulation_name(simulations[sim])

    for planner in range(len(planners)):
        planners[planner] = clean_method_name(planners[planner])
    # fig = scripts.visuals.comparitive_boxplot(simulation_data[data_names[0]], data_names[0], planners, "Planner", simulations)
    fig = scripts.visuals.comparative_boxplot_simulations_planners(simulation_data, data_names[0], simulations, planners)
    ax = fig.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel("Task Duration (s)")
    scripts.visuals.save_figure_as(fig, figure_folder, figure_base_name + "_duration", ratio=n_planners*0.25/1.5)

    # fig = scripts.visuals.comparitive_boxplot(simulation_data[data_names[1]], data_names[1], planners, "Planner", simulations)
    fig = scripts.visuals.comparative_boxplot_simulations_planners(simulation_data, data_names[1], simulations, planners)
    scripts.visuals.save_figure_as(fig, figure_folder, figure_base_name + "_infeasibility", ratio=n_planners*0.25/2)

def debug_visual(simulation, planner_name):
    data, experiment_data, metrics, figure_folder = load_data(simulation, planner_name)
    import matplotlib.pyplot as plt

    assert planner_name == "GMPCC" or planner_name == "GMPCCNO", "Can only use GMPCC for this analysis"

    # Verify active constraints
    signal_list = []
    for p in range(int(metrics['num paths']['max'])+1):
        signal_list.append('active_constraints_' + str(p))

    # scripts.visuals.compare_signals(experiment_data, signal_list, highlight_signal_list=[])
    scripts.visuals.plot_multiple_signals(experiment_data, signal_list, "Iteration", "Active Constraints", use_subplots=True)
    plt.show()


def visualize_intrusion(simulations, planners, base_figure_name):
    ata_folder, metric_folder, figure_folder = get_folder_names()

    simulations *= len(planners)  # Repeat the simulations for each planner
    def duplicate(testList, n):
        return [ele for ele in testList for _ in range(n)]
    n_planners = len(planners)
    planners = duplicate(planners, int(len(simulations) / len(planners)))  # Duplicate planners repeating the same planner N times
    assert len(planners) == len(simulations), "after copying there should be equal number of entries in simulations and planners"

    data_names = ["Intrusion"]
    data_functions = [lambda experiment_data, metrics: merge_experiment_data(experiment_data, "max_intrusion")]

    # Load the data from a cache if it exists (this prevents the slow loading of all planner data)
    if not reload_cache:
        simulation_data, cache_exists = load_from_cache(simulations, planners)
    else:
        cache_exists = False

    if not cache_exists:
        simulation_data = merge_simulation_data(simulations, planners, data_names, data_functions)
        save_in_cache(simulations, planners, simulation_data)

    for sim in range(len(simulations)):
        simulations[sim] = clean_simulation_name(simulations[sim])

    for planner in range(len(planners)):
        planners[planner] = clean_method_name(planners[planner])
    # fig = scripts.visuals.comparitive_boxplot(simulation_data[data_names[0]], data_names[0], planners, "Planner", simulations)
    fig = scripts.visuals.comparative_boxplot_simulations_planners(simulation_data, data_names[0], simulations, planners)
    ax = fig.gca()
    ax.set_xlabel("Intrusion (m)")
    scripts.visuals.save_figure_as(fig, figure_folder, f"{base_figure_name}_intrusion", ratio=n_planners * 0.25 / 1.5)

def plot_trajectories(simulation, planner_name):
    data, experiment_data, metrics, figure_folder = load_data(simulation, planner_name)

    # Clean the current files for this experiment
    save_folder = figure_folder + "../analysis/" + simulation + "-" + planner_name
    try:
        shutil.rmtree(save_folder)
    except FileNotFoundError:
        pass

    random.seed(0)

    # Save an animation for all collision cases and some (randomly picked) success cases
    for e in tqdm(range(metrics["num experiments"])):
        if metrics["severe collisions"][e]:
            if metrics["passive collisions"][e]:
                scripts.visuals.animate_trajectories(experiment_data, e, simulation, planner_name, subfolder="passive_collisions/")
            else:
                scripts.visuals.animate_trajectories(experiment_data, e, simulation, planner_name, subfolder="active_collisions/")
        elif e % 25 == 0:
            scripts.visuals.animate_trajectories(experiment_data, e, simulation, planner_name, subfolder="success/")


def experimental_figures_diego(experiment, planner):
    filename = experiment + "_" + planner
    data_folder, metric_folder, figure_folder = get_folder_names()
    data_path = data_folder + filename + ".txt"

    metrics = dict()

    experiment_data = load_diego_ros_data(data_path)
    figure_folder += f"{experiment}/"
    figure_name = lambda i: f"{experiment}_{planner}_trajectories_{i}"

    margin = 1.5
    # X and Y flipped so that figures match the camera rotation
    MIN_X = -2.5 - margin
    MAX_X = 3.7 + margin
    MAX_Y = 3.5 + margin
    MIN_Y = -4.0 - margin
    skip_first = 0
    for idx, e in enumerate(experiment_data):
        fig = plt.figure()
        ax = plt.gca()

        # Plot the robot and obstacles
        scripts.visuals.plot_experimental_trajectories(ax, e["vehicle_pose"], t_start=skip_first, color='C0', text='Robot')
        for v in range(e['num obstacles']):
            scripts.visuals.plot_experimental_trajectories(ax, e[f"disc_{v + 1}_pose"], color='C2', t_start=skip_first, radius=0.4, text=str(v + 1))

        # Figure settings
        plt.xlim([MIN_X, MAX_X])
        plt.ylim([MIN_Y, MAX_Y])
        scripts.visuals.save_figure_as(fig, figure_folder, figure_name(idx), ratio=1., save_png=False)

def filter_obstacles_that_are_standing_still(metrics, experiment):
    add = "disc_0_pose" not in experiment.keys()
    for v in range(metrics['num obstacles']):
        cur_data = experiment[f"disc_{v + add}_pose"]

        # Detect standing still
        if(len(cur_data) == 0):
            continue
        if dist(cur_data[0, :], cur_data[-1, :]) < 1.:
            experiment[f"disc_{v + add}_pose"] = list()
            print(f"removing obstacle {v + add} (standing still)")
            continue

        # for t in range(1, cur_data.shape[0]):
        #     if dist(cur_data[t, :], np.array([0, 0])) > 50:
        #         cur_data[t, :] = cur_data[t-1, :]
        # experiment[f"disc_{v + add}_pose"] = cur_data

        experiment[f"disc_{v + add}_pose"] = np.array([np.array([pos[0], pos[1]]) for pos in experiment[f"disc_{v + add}_pose"] if math.sqrt(pos[0]**2 + pos[1]**2) < 50.])


def experimental_figures(experiment, planner):
    data_folder, metric_folder, figure_folder = get_folder_names()
    data, experiment_data, metrics, figure_folder = load_data(experiment, planner, fix_jumps=True)
    figure_folder += f"{experiment}/"
    figure_name = lambda i : f"{experiment}_{planner}_trajectories_{i}"

    margin = 1.0
    # X and Y flipped so that figures match the camera rotation
    MIN_X = -3.5 - margin
    MAX_X = 5.0 + margin
    MAX_Y = 3.5 + margin
    MIN_Y = -4.0 - margin

    # For overlay
    # MIN_X = -3.5 - margin
    # MAX_X = 5.0 + margin
    # MIN_X += ((133. - 18) / 475.36) * (MAX_X - MIN_X)
    # MAX_X -= ((142.32 + 18) / 475.36) * (MAX_X - MIN_X)
    # MAX_Y = 3.5 + margin
    # MIN_Y = -4.0 - margin
    # MIN_Y += (109.89 / 429.) * (MAX_Y-MIN_Y)
    # MAX_Y -= (145. / 429.) * (MAX_Y-MIN_Y)

    skip_first = 65 #75

    font = {'family': 'serif', 'serif': ["Computer Modern Serif"], # 'weight' : 'bold',
        'size': 24}
    plt.rc('font', **font)

    for idx, e in enumerate(experiment_data):

        fig = plt.figure()
        ax = plt.gca()

        # Plot the robot and obstacles
        # scripts.visuals.plot_experimental_trajectories(ax, e["vehicle_pose"], t_start=skip_first, color='C0', text='Robot')
        scripts.visuals.plot_experimental_trajectories(ax, e["vehicle_pose"], t_start=skip_first, color='C0', text='')
        add = "disc_0_pose" not in e.keys()
        filter_obstacles_that_are_standing_still(metrics, e)
        for v in range(metrics['num obstacles']):
            # e[f"disc_{v + add}_pose"] = np.array([np.array([pos[0], pos[1]]) for pos in e[f"disc_{v + add}_pose"] if math.sqrt(pos[0]**2 + pos[1]**2) < 50.])
            if len(e[f"disc_{v + add}_pose"]) > 0:
                scripts.visuals.plot_experimental_trajectories(ax, e[f"disc_{v + add}_pose"],
                                                               color='C2', t_start=skip_first, radius=0.4,
                                                               text='', show_trace=True)

                # color='C2', t_start=skip_first, radius=0.4, text=str(v+1), show_trace=False)

        # Figure settings
        plt.xlim([MIN_X, MAX_X])
        plt.ylim([MIN_Y, MAX_Y])
        # plt.xlabel('X [m]')
        # plt.ylabel('Y [m]')
        plt.axis('off')

        scripts.visuals.save_figure_as(fig, figure_folder, figure_name(idx), ratio=(MAX_Y-MIN_Y)/(MAX_X-MIN_X), save_png=True, transparent=True)


        # plt.show()


if __name__ == '__main__':
    run_what = sys.argv[1]
    reload_cache = True # Set to true to not use the cache and load new data

    if run_what == "plot":  # Make a GIF of a particular experiment
        assert len(sys.argv) > 3
        simulation = sys.argv[2]
        method = sys.argv[3]

        plot_trajectories(simulation, method)
    if run_what == "experimental_figures_diego":  # Make a GIF of a particular experiment
        assert len(sys.argv) > 3
        experiment = sys.argv[2]
        planner = sys.argv[3]
        experimental_figures_diego(experiment, planner)
    if run_what == "experimental_figures":  # Make a GIF of a particular experiment
        assert len(sys.argv) > 3
        experiment = sys.argv[2]
        planner = sys.argv[3]
        experimental_figures(experiment, planner)

    elif run_what == "debug":
        simulation = sys.argv[2]
        planner_name = sys.argv[3]
        debug_visual(simulation, planner_name)
    elif run_what == "deterministic_boxplots":
        simulations = ["random_fast-4_straight", "random_fast-8_straight", "random_fast-12_straight", "random_fast-16_straight"]
        planners = ["frenet-planner", "LMPCC", "GMPCCNO", "GMPCC"]
        boxplot_of_multiple_simulations_and_planners(simulations, planners, "deterministic")
    elif run_what == "uncertain_boxplots":
        simulations = ["1-random_social-12_uncertain", "01-random_social-12_uncertain", "001-random_social-12_uncertain"]
        planners = ["LMPCC", "GMPCC"]
        boxplot_of_multiple_simulations_and_planners(simulations, planners, "uncertain", True)
    elif run_what == "pedsim_interactive_boxplots":
        simulations = ["interactive_random_social-4_corridor", "interactive_random_social-8_corridor", "interactive_random_social-12_corridor"]
        planners = ["LMPCC", "GMPCCNO", "GMPCC","Guidance-MPCCNO", "frenet-planner"]
        boxplot_of_multiple_simulations_and_planners(simulations, planners, "pedsim_interactive", False)
    elif run_what == "analyze_intrusion":
        simulations = ["random_social-16_corridor"]
        planners = ["LMPCC", "GMPCCNO", "GMPCC"]
        visualize_intrusion(simulations, planners, "notinteractive")
