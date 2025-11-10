#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import warnings
import numpy as np

import matplotlib.pyplot as plt
from scripts.experiments.static_comparison.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data


def get_next_largest(value_list, max_val):
    for index, value in enumerate(value_list):
        if abs(value - max_val) < 1:
            value_list[index] = 0.

    print(f"Max value: {np.max(value_list)}, index: {np.argmax(value_list)}")
    return np.argmax(value_list), np.max(value_list), value_list

def process(simulation, planner_name):
    print(f'Generating trajectory comparison figure (LMPCC versus GMPCC)')
    experiment_data, metrics, figure_folder = visuals.load_simulation_and_planner(simulation, planner_name, filter_faulty=False)

    # PLOT TRAJECTORIES
    # for t in range(0, 70, 5):
    ellipsoid_objective = load_ros_data.merge_experiment_data(experiment_data, "objective_2")[:-2]
    # ellipsoid_objective[ellipsoid_objective is np.NAN] = 0.

    moments = []
    max_value = 1e9
    while max_value != 0.:
        new_moment, max_value, ellipsoid_objective = get_next_largest(ellipsoid_objective, max_value)
        moments.append(new_moment)

    for index in range(len(moments)):
        fig = visuals.plot_plan(experiment_data, moments[index], ["solver0_plan", "solver1_plan", "vehicle_plan_"], ["solver2_plan"])
        plt.legend(["GMPCC", "_nolegend_", "_nolegend_", "LMPCC"], fontsize=16, loc="upper left")
        plt.grid()
        # fig.canvas.manager.full_screen_toggle()  # toggle fullscreen mode
        plt.xlim([0, 6.75])
        plt.ylim([-2, 2])

        visuals.save_figure_as(fig, figure_folder + simulation, f'trajectory_comparison{index}.pdf', ratio=0.8)

    # PLOT THE OBJECTIVE VALUES
    fig = visuals.compare_signals(
        experiment_data, ["objective_0", "objective_1", "objective"], ["objective_2"], filter=lambda value: value == -1
        )

    plt.legend(['GMPCC Guidance 0', 'GMPCC Guidance 1', 'GMPCC', 'LMPCC'], fontsize=16, loc="upper right", ncol=2)
    # plt.ylim([10, 25])
    plt.autoscale(fig, axis='y', tight=False)
    plt.xlim([0, 50])
    visuals.save_figure_as(fig, figure_folder + simulation, 'objective_values.pdf', ratio=0.4)

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

