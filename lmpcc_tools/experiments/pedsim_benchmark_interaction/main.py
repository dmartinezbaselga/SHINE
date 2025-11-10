#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import matplotlib.pyplot as plt
from scripts.experiments.pedsim_benchmark_interaction.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data

def process(is_original_planner=False):
    print(f'Comparing performance of GMPCC under varying number of paths')
    paths = list(range(0, 7)) # ##

    # duration_data = []
    # collision_data = []
    # infeasible_data = []
    # paths_exist = []
    # for path in paths:
    #     if is_original_planner:
    #         simulation = f'{path}pathsO-random_fast-16_straight'
    #     else:
    #         simulation = f'{path}paths-random_fast-16_straight'
    #
    #     try:
    #         experiment_data, metrics, figure_folder = visuals.load_simulation_and_planner(simulation, "GMPCC", filter_faulty=True)
    #         duration_data.append(load_ros_data.merge_experiment_data(experiment_data, "metric_duration"))
    #         # collision_data.append(load_ros_data.merge_experiment_data(experiment_data, "metric_collisions"))
    #         print(path)
    #         collision_data.append(metrics['num severe collisions'])
    #         infeasible_data.append(metrics['num infeasible'])
    #         paths_exist.append(path)
    #     except FileNotFoundError:
    #         pass
    #
    # fig = visuals.signal_boxplot(duration_data,"Task Duration", paths_exist, "Number of Trajectories")
    # plt.ylim([12.276525878906252, 18.2765])#24.873247917330996])
    # print(plt.gca().get_ylim())
    #
    # fig_infeasible = visuals.signal_boxplot(infeasible_data,"Infeasible", paths_exist, "Number of Trajectories")
    # plt.ylim([0, 20])
    #
    # if is_original_planner:
    #     visuals.save_figure_as(fig, figure_folder, f'path_comparison_wo.pdf', ratio=0.3)
    #     visuals.save_figure_as(fig_infeasible, figure_folder, f'path_comparison_infeasible_wo.pdf', ratio=0.3)
    # else:
    #     visuals.save_figure_as(fig, figure_folder, f'path_comparison.pdf', ratio=0.3)
    #     visuals.save_figure_as(fig_infeasible, figure_folder, f'path_comparison_infeasible.pdf', ratio=0.3)
    #
    # fig = plt.figure()
    # ax = fig.gca()
    # fig = visuals.subplot_signal(ax, collision_data)
    # ax.set_xlabel('Paths')
    # ax.set_ylabel('Collisions')
    # print(collision_data)

    # plt.show()
    #     plt.legend(['GMPCC (incl. LMPCC)', 'LMPCC'], fontsize=16, loc="upper right", ncol=2)
    #     # plt.ylim([10, 25])
    #     plt.autoscale(fig, axis='y', tight=False)
    #     # plt.xlim([0, 50])

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == "parameters":
        set_parameters(sys.argv[2:])
    elif stage == "restore":
        parameters.remove_config()
    elif stage == "process":
        process(is_original_planner=(len(sys.argv) > 2 and sys.argv[2]=='True'))

