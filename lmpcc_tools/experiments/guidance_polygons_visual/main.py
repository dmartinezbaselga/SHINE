#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import warnings

import matplotlib.pyplot as plt
from scripts.experiments.guidance_polygons_visual.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data

def process(simulation, planner_name):
    print(f'Generating trajectory comparison figure (LMPCC versus GMPCC)')
    experiment_data, metrics, figure_folder = visuals.load_simulation_and_planner(simulation, planner_name, filter_faulty=False)

    t = 130

    fig = visuals.plot_linearized_constraint_regions(experiment_data, t, ["solver1_plan", "solver0_plan"])
    # plt.legend(["Plan 1", "Plan 2"], fontsize=16, loc="upper center",ncol=2, bbox_to_anchor=(0.47, 1.25))

    axs = fig.axes
    for ax in axs:
        ax.set_xlim([0, 6.5])
        ax.set_ylim([-1.0, 1.0])

    visuals.save_figure_as(fig, figure_folder + simulation, f'guidance_polygons_{simulation}.pdf', ratio=0.8)

    # fig2 = visuals.plot_linearized_constraint_regions(experiment_data, t, ["solver0_plan"], color_start=0)
    # plt.xlim([0, 6.5])
    # plt.ylim([-1.0, 1.0])
    #
    # fig3 = visuals.plot_linearized_constraint_regions(experiment_data, t, ["solver1_plan"], color_start=1)
    # plt.xlim([0, 6.5])
    # plt.ylim([-1.0, 1.0])
    # fig = visuals.compare_signals(experiment_data, ["objective_0", "objective_1"],
    #                               filter=lambda value: value == -1)
    plt.show()


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

