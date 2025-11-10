#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scripts.experiments.path_comparison.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data
from scripts.helpers import load_from_cache, save_in_cache, get_folder_names
from scripts.compare import clean_method_name, clean_simulation_name

def process(is_original_planner=False):
    print(f'Comparing performance of GMPCC under varying number of paths')
    data_folder, metric_folder, figure_folder = get_folder_names()
    figure_folder += "path_comparison/"

    if is_original_planner:
        paths = list(range(0, 7)) # ##
    else:
        paths = list(range(1, 7))

    data_names = ["Task Duration", "Infeasible"]
    data_functions = [lambda experiment_data, metrics: load_ros_data.merge_experiment_data(experiment_data, "metric_duration"), lambda experiment_data, metrics: metrics['num infeasible']]

    # Define simulations and the planner used
    simulations = []
    for path in paths:
        if is_original_planner:
            simulations.append(f'{path}pathsO-random_fast-16_straight')
        else:
            simulations.append(f'{path}paths-random_fast-16_straight')

    def duplicate(testList, n):
        return [ele for ele in testList for _ in range(n)]
    planners = duplicate(["GMPCC"], len(paths))

    # Load data if it exists in the cache, or otherwise load from simulation data
    simulation_data, cache_exists = load_from_cache(simulations, planners)
    if not cache_exists:
        simulation_data = load_ros_data.merge_simulation_data(simulations, planners, data_names, data_functions)
    save_in_cache(simulations, planners, simulation_data)

    # Clean the simulation / method names for visualization
    for sim in range(len(simulations)):
        simulations[sim] = paths[sim]

    for planner in range(len(planners)):
        planners[planner] = clean_method_name(planners[planner])

    fig_duration = visuals.simulations_boxplot(simulation_data, "Task Duration", "Number of Trajectories", simulations)
    plt.xlim([13, 20])#
    # print(plt.gca().get_ylim())

    fig_infeasible = visuals.simulations_boxplot(simulation_data, "Infeasible", "Number of Trajectories", simulations)
    plt.xlim([0, 20])
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if is_original_planner:
        visuals.save_figure_as(fig_duration, figure_folder, 'path_duration_wo', ratio=0.3)
        visuals.save_figure_as(fig_infeasible, figure_folder, 'path_infeasible_wo', ratio=0.3)
    else:
        visuals.save_figure_as(fig_duration, figure_folder, 'path_duration', ratio=0.3)
        visuals.save_figure_as(fig_infeasible, figure_folder, 'path_infeasible', ratio=0.3)

    # fig = plt.figure()
    # ax = fig.gca()
    # fig = visuals.subplot_signal(ax, collision_data)
    # ax.set_xlabel('Paths')
    # ax.set_ylabel('Collisions')

    # print(collision_data)

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == "parameters":
        set_parameters(sys.argv[2:])
    elif stage == "restore":
        parameters.remove_config()
    elif stage == "process":
        process(is_original_planner=(len(sys.argv) > 2 and sys.argv[2]=='True'))

