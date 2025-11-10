from scripts import set_parameters as parameters

def set_parameters(method_name):
    if method_name == "UVD":
        return dynamic_uvd()
    elif method_name == "Homology":
        return dynamic_homology()
    else:
        raise IOError("not a valid method")

def dynamic_uvd():
    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 1
    params["recording"]["enable_recording"] = True
    params["recording"]["prepend_name"] = "UVD-"

    params["prm"] = dict()
    params["prm"]["topology_comparison"] = "UVD"  # ms
    params["prm"]["n_paths"] = 7

    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = False
    parameters.write_config(params)
    return params

def dynamic_homology():
    params = dynamic_uvd()
    params["prm"]["topology_comparison"] = "Homology"  # ms
    params["recording"]["prepend_name"] = "Homology-"

    parameters.write_config(params)
    return params