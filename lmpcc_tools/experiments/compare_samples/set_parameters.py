import scripts.set_parameters as parameters

def set_parameters(args):
    num_samples = args[0]

    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 100
    params["recording"]["enable_recording"] = True

    params["recording"]["prepend_name"] = f"{num_samples}samples-"

    params["prm"] = dict()
    params["prm"]["n_samples"] = int(num_samples)

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = True

    parameters.write_config(params)
    return params