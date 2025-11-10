from scripts import set_parameters as parameters

def set_parameters():
    params = parameters.static()

    params["prm"]["selection_weights"] = dict()
    params["prm"]["selection_weights"]["consistency"] = 0.

    params["prm"]["n_samples"] = 2000
    params["prm"]["n_paths"] = 2

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = True

    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 1
    params["recording"]["enable_recording"] = True
    params["recording"]["time_out"] = 10
    parameters.write_config(params)
    return params