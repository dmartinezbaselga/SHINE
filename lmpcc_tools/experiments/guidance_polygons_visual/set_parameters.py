from scripts import set_parameters as parameters
from scripts.experiments.static_comparison.main import set_parameters as set_static_parameters

def set_parameters():
    params = set_static_parameters()
    return params