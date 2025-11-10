import scripts.set_parameters as parameters
import scripts.helpers as helpers

def set_parameters(args):
    helpers.verify_solver_configuration("GMPCC")
    num_paths = args[0]
    use_original_planner = args[1] == 'True'

    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 100
    params["recording"]["enable_recording"] = True

    if use_original_planner:
        params["recording"]["prepend_name"] = f"{num_paths}pathsO-"
    else:
        params["recording"]["prepend_name"] = f"{num_paths}paths-"


    params["prm"] = dict()
    params["prm"]["n_paths"] = int(num_paths)

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = use_original_planner

    parameters.write_config(params)
    return params