# This script was created with ChatGPT :P
# I just corrected some small bugs
# Enjoy !
import copy

import numpy as np
import os
import json

import math

from copy import deepcopy

from scripts.helpers import bcolors, get_folder_names, PROJECT_FOLDER

def load_ros_data(filename, verbose=True):
    if verbose:
        print("Loading data from: {}".format(filename))

    data = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File not found: {filename}")
    with open(filename, 'r') as file:
        for line in file:
            entry = line.split(':')
            if len(entry) < 2:
                continue
            var_name = entry[0].strip()
            if var_name == '-':
                break
            columns_per_var, rows_per_var = map(int, entry[1].split())
            data[var_name] = np.zeros((rows_per_var, columns_per_var))
            data[var_name] = np.genfromtxt(file, delimiter=' ', max_rows=rows_per_var)
            data[var_name] = data[var_name].reshape(rows_per_var, columns_per_var)
    return data


def split_ros_data(all_data, remove_first=True, debug=False, num_experiments=int(1e6), filter_faulty=True):
    split_data = []
    start_it = 0
    for reset in all_data["reset"]:
        split_data.append(dict())
        experiment_index = len(split_data) - 1
        for key, value in all_data.items():
            # split_data[-1][key] = np.zeros((int(reset) - start_it, 1))
            if len(value) == len(all_data["reset"]):
                split_data[experiment_index][key] = value[experiment_index][0]
            else:
                split_data[experiment_index][key] = value[start_it:int(reset)]

        start_it = deepcopy(int(reset))

    # TEST: Overlap
    if debug:
        for i in range(1, len(split_data)):
            prev_exp = split_data[i-1]
            cur_exp = split_data[i]
            assert prev_exp["runtime_control_loop"][-1] != cur_exp["runtime_control_loop"][0]

    if remove_first:
        remove_experiment(split_data, 0)        # Remove the first experiment because setup is still running here

    if filter_faulty:
        filter_faulty_experiments(split_data)       # Filter experiments with issues in the setup
    split_data = split_data[0:num_experiments]  # Cap the number of experiments!

    return split_data



def convert_diego_to_lmpcc(data):

    for e in data:
        e['num obstacles'] = len(e['obs_x'][0])
        # for i in range(len(data['px'])):
        #     data['vehicle_pose'] = np.array([])
        e['vehicle_pose'] = np.vstack([e['px'], e['py']]).transpose()
        for j in range(e['num obstacles']):
            e[f"disc_{j + 1}_pose"] = np.vstack([np.array(e['obs_x'])[:, j], np.array(e['obs_y'])[:, j]]).transpose()

    return data

def load_diego_ros_data(filename, verbose=True):
    if verbose:
        print("Loading data from: {}".format(filename))

    data = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File not found: {filename}")
    with open(filename, 'r') as file:
        experiment_data = []

        cur_experiment = dict()
        for line in file:
            entry = line.split(',')
            if len(entry) == 1 and len(cur_experiment.keys()) > 0:
                experiment_data.append(copy.deepcopy(cur_experiment))
                cur_experiment = dict()
            elif len(entry) > 1:
                var_name = line.split(':')[0].strip()
                if len(line.split('[')) == 1:
                    cur_experiment[var_name] = np.fromstring(line.split(':')[1], sep=', ')
                else:
                    cur_experiment[var_name] = []
                    num_el = len(line.split('[')[1].split(']')[0].split(',')) - 1
                    data_str = line.split(':')[1].split('[')[1:]
                    for str in data_str:
                        res = np.zeros((num_el))
                        elements = str.replace("]", "").replace(",", "").strip().split(" ")
                        for i in range(num_el):
                            if i >= len(elements):
                                res[i] = np.nan
                            else:
                                res[i] = float(elements[i])
                        cur_experiment[var_name].append(res)
    return convert_diego_to_lmpcc(experiment_data)


def remove_experiment(experiment_data, index):
    del experiment_data[index]


def filter_faulty_experiments(experiment_data):

    filtered_experiments = []
    for i, experiment in enumerate(experiment_data):
        vehicle_pose = experiment["vehicle_pose"]

        it_from_start = -1
        for it in range(10, len(vehicle_pose), 10):
            if vehicle_pose[it, 0] > 1.5:
                it_from_start = it
                break
        if it_from_start == -1:
            it_from_start = 1e3

        if it_from_start > 100:
            filtered_experiments.append(i)

    for experiment_idx in sorted(filtered_experiments, reverse=True):
        print(bcolors.WARNING + "\tRemoving faulty experiment (index = " + str(experiment_idx) + ")!" + bcolors.ENDC)
        remove_experiment(experiment_data, index=experiment_idx)


def load_simulation_and_planner(simulation, planner_name, **kwargs):

    data_folder, metric_folder, figure_folder = get_folder_names(PROJECT_FOLDER)

    # metric_folder = os.getcwd() + "/metrics/" + simulation
    metric_path = metric_folder + simulation + "/" + planner_name + ".json"

    filename = simulation + "_" + planner_name
    data_path = data_folder + filename + ".txt"
    data = load_ros_data(data_path, verbose=False)
    experiment_data = split_ros_data(data, **kwargs)

    # print("Loading JSON: {}".format(str(metric_path)))
    with open(metric_path) as f:
        metrics = dict(json.load(f))

    # save_path = os.getcwd() + "/figures/" + simulation + "/"
    os.makedirs(figure_folder, exist_ok=True)

    return experiment_data, metrics, figure_folder


def merge_experiment_data(experiment_data, data_key):
    merged_data = np.ones((0, 1))
    for experiment in experiment_data:
        merged_data = np.vstack([merged_data, experiment[data_key]])

    merged_data = [float(y) for y in merged_data]
    return merged_data


def merge_simulation_data(simulation_list, planner_list, data_names, data_functions):
    simulation_data = dict()

    for idx, data_function in enumerate(data_functions):
        simulation_data[data_names[idx]] = []

    for s, simulation in enumerate(simulation_list):
        try:
            print("Loading data for simulation " + simulation + " and planner " + planner_list[s])
            experiment_data, metrics, figure_folder = load_simulation_and_planner(simulation, planner_list[s], filter_faulty=False)

            for idx, data_function in enumerate(data_functions):
                simulation_data[data_names[idx]].append(data_function(experiment_data, metrics))
        except FileNotFoundError:
            print("data for simulation " + simulation + " does not exist")

    return simulation_data



