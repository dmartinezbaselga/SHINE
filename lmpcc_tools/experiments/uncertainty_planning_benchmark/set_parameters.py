import scripts.set_parameters as parameters
import scripts.helpers as helpers

def set_parameters(args):
    assert len(args) >= 3
    helpers.verify_solver_configuration("GMPCC-Gaussian")

    use_original_planner = args[0] == 'True'

    params = dict()
    params["recording"] = dict()
    params["recording"]["number_of_experiments"] = 210
    params["recording"]["enable_recording"] = True
    params["recording"]["prepend_name"] = str(args[1]).split(".")[1] + "-"


    params["guidance"] = dict()
    params["guidance"]["constraints"] = dict()
    params["guidance"]["constraints"]["add_original_problem"] = use_original_planner

    params["prm"] = dict()
    params["prm"]["n_paths"] = int(args[2])
    params["prm"]["predictions_are_constant_velocity"] = False

    if int(args[2]) == 0 and use_original_planner:
        params["prm"]["n_samples"] = 10

    params["scenarios"] = dict()
    params["scenarios"]["manual_noise"] = 0.3
    params["scenarios"]["safe_sampling"] = dict()
    params["scenarios"]["safe_sampling"]["risk"] = float(args[1])

    params["pedestrian_simulator"] = dict()
    params["pedestrian_simulator"]["pedestrians"] = dict()
    params["pedestrian_simulator"]["pedestrians"]["process_noise"] = [0.3, 0.3]
    params["pedestrian_simulator"]["pedestrians"]["constant_velocity_predictions"] = False
    params["pedestrian_simulator"]["pedestrians"]["interaction"] = True

    parameters.write_config(params)
    return params