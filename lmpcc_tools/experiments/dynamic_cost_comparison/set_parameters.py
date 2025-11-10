from scripts import set_parameters as parameters

def set_parameters():
    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 10
    params["recording"]["enable_recording"] = True
    params["recording"]["time_out"] = 30
    params["recording"]["prepend_name"] = "cost-"

    params["prm"] = dict()
    params["prm"]["selection_weights"] = dict()
    params["prm"]["selection_weights"]["consistency"] = 0.

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = True
    parameters.write_config(params)
    return params