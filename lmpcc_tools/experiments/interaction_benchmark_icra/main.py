#!/usr/bin/env python3

import sys, os
sys.path.append("..")

import matplotlib.pyplot as plt
from scripts.experiments.interaction_benchmark_icra.set_parameters import set_parameters
from scripts import set_parameters as parameters
from scripts import visuals, load_ros_data

def process(is_original_planner=False):
    pass

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == "parameters":
        set_parameters()
    elif stage == "restore":
        parameters.remove_config()
    elif stage == "process":
        process()

