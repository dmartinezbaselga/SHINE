import scripts.set_parameters as parameters
import scripts.helpers as helpers

import os

def set_parameters():
    helpers.verify_solver_configuration("ICRA-2023")

    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 210
    params["recording"]["enable_recording"] = True
    params["recording"]["prepend_name"] = "interactive_" # To distinguish from the non interactive scenario

    params["guidance"] = dict()
    params["guidance"]["use_as_reference"] = True
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = False

    params["pedestrian_simulator"] = dict()
    params["pedestrian_simulator"]["pedestrians"] = dict()
    params["pedestrian_simulator"]["pedestrians"]["constant_velocity_predictions"] = False
    params["pedestrian_simulator"]["pedestrians"]["interaction"] = True

    params["weights"] = dict()
    params["weights"]["contour"] = 0.5 # 10x higher!

    params["prm"] = dict()
    params["prm"]["topology_comparison"] = "Homology"
    params["prm"]["n_samples"] = 100
    params["prm"]["predictions_are_constant_velocity"] = False

    params["prm"]["spline_optimization"] = dict()
    params["prm"]["spline_optimization"]["enable"] = True

    params["prm"]["selection_weights"] = dict()
    params["prm"]["selection_weights"]["consistency"] = 2. # Value is used in a different manner for the ICRA version, so should be larger


    parameters.write_config(params)
    return params