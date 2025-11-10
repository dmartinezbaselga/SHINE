import scripts.set_parameters as parameters

def set_parameters(args):
    use_original_planner = args[0] == 'True'

    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 250
    params["recording"]["enable_recording"] = True

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = use_original_planner

    params["pedestrian_simulator"] = dict()
    params["pedestrian_simulator"]["pedestrians"] = dict()
    params["pedestrian_simulator"]["pedestrians"]["constant_velocity_predictions"] = True

    params["prm"] = dict()
    params["prm"]["predictions_are_constant_velocity"] = True

    if len(args) == 2:
        params["prm"] = dict()
        params["prm"]["n_paths"] = int(args[1])

        if int(args[1]) == 0 and use_original_planner:
            params["prm"]["n_samples"] = 10

    parameters.write_config(params)
    return params