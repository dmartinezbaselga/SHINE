#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import matplotlib.pyplot as plt
from scripts.experiments.compare_samples.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data
from scripts import metrics as metrics_scripts

def process():
    print(f'Comparing performance of GMPCC under varying number of samples')
    samples = [10, 20, 30, 40, 50, 100, 200]

    duration_data = []
    collision_data = []
    infeasible_data = []
    runtime_data = []
    guidance_runtime_data = []
    samples_exist = []
    for n_samples in samples:
        simulation = f'{n_samples}samples-random_fast-16_straight'

        try:
            experiment_data, metrics, figure_folder = visuals.load_simulation_and_planner(simulation, "GMPCC", filter_faulty=True)
            
            metrics["guidance runtime"] = dict()
            metrics_scripts.add_mean_std_max_min(metrics["guidance runtime"], metrics_scripts.merge_experiment_data(experiment_data, "runtime_modules"))
            print(n_samples)
            duration_data.append(load_ros_data.merge_experiment_data(experiment_data, "metric_duration"))
            # collision_data.append(load_ros_data.merge_experiment_data(experiment_data, "metric_collisions"))
            collision_data.append(metrics['num severe collisions'])
            infeasible_data.append(metrics['num infeasible'])
            # runtime_data.append(metrics['guidance runtime']['mean'])
            runtime_data.append(metrics_scripts.merge_experiment_data(experiment_data, "runtime_control_loop"))
            guidance_runtime_data.append(metrics_scripts.merge_experiment_data(experiment_data, "runtime_modules"))
            samples_exist.append(n_samples)

        except FileNotFoundError:
            pass

    fig_taskduration = visuals.signal_boxplot(duration_data,"Task Duration", samples_exist, "Samples")
    plt.ylim([12.276525878906252, 18.2765])#24.873247917330996])
    print(plt.gca().get_ylim())

    fig_infeasible = visuals.signal_boxplot(infeasible_data,"Infeasible", samples_exist, "Samples")
    plt.ylim([0, 20])    
        
    fig_runtime = visuals.signal_boxplot(runtime_data,"Runtime", samples_exist, "Samples", do_strip_plot=False)
    plt.ylim([0., 0.05])
    print("runtime y:{}".format(plt.gca().get_ylim()))

    fig_guidance_runtime = visuals.signal_boxplot(guidance_runtime_data,"Guidance Runtime", samples_exist, "Samples", do_strip_plot=False)
    plt.ylim([0., 0.05])
    print("guidanceruntime y:{}".format(plt.gca().get_ylim()))

    # plt.ylim([0, 20])
    fig_collisions = plt.figure()
    plt.plot(samples, collision_data)
    # plt.ylim([0.0, 0.05])
    # plt.xlim([samples[0], samples[-1]])
    
    # fig, ax = plt.figure()
    # visuals.subplot_signal(ax, runtime_data)

    visuals.save_figure_as(fig_taskduration, figure_folder, f'sample_comparison/task_duration.pdf', ratio=0.3)
    visuals.save_figure_as(fig_infeasible, figure_folder, f'sample_comparison/infeasible.pdf', ratio=0.3)
    visuals.save_figure_as(fig_runtime, figure_folder, f'sample_comparison/runtime.pdf', ratio=0.3)
    visuals.save_figure_as(fig_guidance_runtime, figure_folder, f'sample_comparison/guidance_runtime.pdf', ratio=0.3)
    visuals.save_figure_as(fig_collisions, figure_folder, f'sample_comparison/collisions.pdf', ratio=0.3)

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
        process()

