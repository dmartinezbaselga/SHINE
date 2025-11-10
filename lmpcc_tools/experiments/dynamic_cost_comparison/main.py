#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import warnings

import matplotlib.pyplot as plt
from scripts.experiments.dynamic_cost_comparison.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data

def process(simulation, planner_name):
    print(f'Comparing cost of LMPCC in GMPCC and generating figures')
    experiment_data, metrics, figure_folder = visuals.load_simulation_and_planner(simulation, planner_name, filter_faulty=False)
    LMPCC_name = f"objective_{experiment_data[0]['original_planner_id'][0][0]:.0f}"

    for i, experiment in enumerate(experiment_data):
        # PLOT THE OBJECTIVE VALUES
        fig = visuals.compare_signals([experiment_data[i]], ["gmpcc_objective"], [LMPCC_name])  # ,
        # filter=lambda value: value == -1)

        plt.legend(['GMPCC (incl. LMPCC)', 'LMPCC'], fontsize=16, loc="upper right", ncol=2)
        # plt.ylim([10, 25])
        plt.autoscale(fig, axis='y', tight=False)
        # plt.xlim([0, 50])
        visuals.save_figure_as(fig, figure_folder + simulation, f'dynamic_cost_comparison{i}.pdf', ratio=0.4)

    # for i, experiment in enumerate(experiment_data):  #     fig, axs = plt.subplots(2, tight_layout=True)  #     # visuals.subplot_signal(axs[0], [experiment_data[i]], "objective_0", color='C1', linewidth=1)  #     # visuals.subplot_signal(axs[1], [experiment_data[i]], "objective_1", color='C1', linewidth=1)  #     # visuals.subplot_signal(axs[2], [experiment_data[i]], "objective_2", color='C1', linewidth=1)  #     # visuals.subplot_signal(axs[3], [experiment_data[i]], "objective_3", color='C1', linewidth=1)  #     visuals.subplot_signal(axs[0], [experiment_data[i]], "gmpcc_objective", color='C0', linewidth=3)  #     visuals.subplot_signal(axs[1], [experiment_data[i]], LMPCC_name, color='C3', linewidth=2)

    # plt.show()


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == "parameters":
        set_parameters()
    elif stage == "restore":
        parameters.remove_config()
    elif stage == "process":
        assert len(sys.argv) >= 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            process(sys.argv[2], sys.argv[3])

