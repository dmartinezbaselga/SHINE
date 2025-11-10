# %% [markdown]
# # Online Adaptive Forecasting

# %% [markdown]
# **NOTE:** For some reason, you might see errors such as
# ```
# RuntimeError: DataLoader worker (pid 1730479) is killed by signal: Aborted.
# ```
# when running some of these cells (really only ones that loop over a DataLoader object). I'm not sure why this happens... it might relate to keyboard interrupting a multiprocessing call? Thankfully, (1) restarting the kernel will fix it, and (2) running this code in a script works without issues.


import re
import os
import glob
import json
import pickle
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import trajectron.visualization as visualization
import trajdata.visualization.vis as trajdata_vis
import torch.distributed as dist


from torch import optim, nn
from torch.utils import data
from tqdm.notebook import tqdm
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron
from trajectron.model.mgcvae import MultimodalGenerativeCVAE
from trajectron.model.trajectron_guidance_Hcost import TrajectronGuidanceHCost, EncoderWithH
from trajectron.utils.comm import all_gather
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Final, List, Optional, Union
from trajdata import UnifiedDataset, AgentType, AgentBatch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %%
# Change this to suit your computing environment and folder structure!

TRAJDATA_CACHE_DIR: Final[str] = "/home/diego/.unified_data_cache"
ETH_UCY_RAW_DATA_DIR: Final[str] = "/home/diego/datasets/eth_ucy_peds"

# %%
AXHLINE_COLORS = {
    "Base": "#DD9787",
    "K0": "#A6C48A",
    "Oracle": "#BCB6FF"
}

SEABORN_PALETTE = {
    "Finetune": "#AA7C85",
    "K0+Finetune": "#2D93AD",
    "Ours": "#67934D",
    
    "Base": "#DD9787",
    "K0": "#A6C48A",
    "Oracle": "#BCB6FF"
}

# %%
def load_model(model_dir: str, device: str, epoch: int = 10, custom_hyperparams: Optional[Dict] = None):
    while epoch > 0:
        save_path = Path(model_dir) / f'model_registrar-{epoch}.pt'
        if save_path.is_file():
            break
        epoch -= 1

    model_registrar = ModelRegistrar("", device)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
        
    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron = TrajectronGuidanceHCost(trajectron, model_registrar)
    trajectron.set_environment()
    trajectron.set_annealing_params()
    model_registrar_new = ModelRegistrar(model_dir, hyperparams["device"])
    trajectron.model_registrar = model_registrar_new
    for key in trajectron.node_models_dict.keys():
        if isinstance(trajectron.node_models_dict[key], MultimodalGenerativeCVAE):
            trajectron.node_models_dict[key] = EncoderWithH(trajectron.node_models_dict[key], trajectron.n_topologies, model_registrar_new)


    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    return trajectron, hyperparams

def collect_metrics(eval_dataloader, rank, model):
     with torch.no_grad():
        # Calculate evaluation loss
        eval_perf = defaultdict(lambda: defaultdict(list))

        batch: AgentBatch
        for batch in tqdm(
            eval_dataloader,
            # ncols=80,
            # unit_scale=dist.get_world_size(),
            # disable=(rank > 0),
            desc=f"Epoch Eval",
        ):
            eval_results: Dict[
                AgentType, Dict[str, torch.Tensor]
            ] = model.predict_and_evaluate_batch(
                batch, update_mode=UpdateMode.BATCH_FROM_PRIOR
            )
            for agent_type, metric_dict in eval_results.items():
                for metric, values in metric_dict.items():
                    eval_perf[agent_type][metric].append(values.cpu().numpy().mean())
        # if torch.cuda.is_available() and dist.get_world_size() > 1:
        #     gathered_values = all_gather(eval_perf)
        #     if rank == 0:
        #         eval_perf = []
        #         for eval_dicts in gathered_values:
        #             eval_perf.extend(eval_dicts)
        # print("Gathered values: ", gathered_values)
        eval_final_results = dict()
        for agent_type, metric_dict in eval_perf.items():
                for metric, values in metric_dict.items():
                    eval_final_results[metric] = np.array(values).mean()
        eval_final_results["accuracy"] = np.average(eval_perf[2]["accuracy"], weights=eval_perf[2]["scenarios_homologies"])
        eval_final_results["scenarios_homologies"] = np.sum(eval_perf[2]["scenarios_homologies"])
        eval_final_results["failed_scenarios"] = np.sum(eval_perf[2]["failed_scenarios"])
        eval_final_results["total_scenarios"] = np.sum(eval_perf[2]["total_scenarios"])
        eval_final_results["mean_hom_scenarios"] = np.average(eval_perf[2]["mean_hom_scenarios"], weights=eval_perf[2]["scenarios_homologies"])
        eval_final_results["loss"] = np.average(eval_perf[2]["loss"], weights=eval_perf[2]["scenarios_homologies"])
        eval_final_results["baseline_accuracy"] = np.average(eval_perf[2]["baseline_accuracy"], weights=eval_perf[2]["scenarios_homologies"])
        print("Results: ", eval_final_results)
        print()
        # if rank == 0:
        #     evaluation.log_batch_errors(
        #         eval_perf,
        #         [
        #             "loss",
        #             "accuracy",
        #             "scenarios_homologies",
        #             "failed_scenarios",
        #             "total_scenarios",
        #             "mean_hom_scenarios",
        #             "baseline_accuracy",
        #             # "ml_ade",
        #             # "ml_fde",
        #             # "min_ade_5",
        #             # "min_ade_10",
        #             # "nll_mean",
        #             # "nll_final",
        #         ],
        #         log_writer,
        #         "eval",
        #         epoch,
        #         curr_iter,
        #     )



# %%
def get_dataloader(
    eval_dataset: UnifiedDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False
):
    return data.DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.get_collate_fn(pad_format="right"),
        pin_memory=False if device == 'cpu' else True,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

# %%
# metrics_list = ['ml_ade', 'ml_fde', 'min_ade_5', 'min_fde_5', 'min_ade_10', 'min_fde_10', 'nll_mean', 'nll_final']

# %% [markdown]
# ### Qualitative Plots

# %%



def test_dataset(eval_scene, model_name):
    # %% 
    peds_datasets = ["eth", "hotel", "univ", "zara1", "zara2"]

    history_sec = 2.8
    prediction_sec = 4.8

    checkpoint = 10

    # train_scene = "zara1"
    # eval_scene = "eth"

    # base_model = glob.glob(f"kf_models/{train_scene}_*_base_*")[0]
    model_path = "experiments/pedestrians/kf_models/"
    base_model = model_path + model_name
    eval_dataset = f"eupeds_{eval_scene}"
    eval_data = f"{eval_dataset}-test_loo"

    base_trajectron, _ = load_model(base_model, device, epoch=checkpoint,
        custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                            "single_mode_multi_sample": False}
    )

    # Load training and evaluation environments and scenes
    attention_radius = defaultdict(lambda: 20.0) # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    batch_eval_dataset = UnifiedDataset(
        desired_data=[eval_data],
        history_sec=(history_sec, history_sec),
        future_sec=(prediction_sec, prediction_sec),
        agent_interaction_distances=attention_radius,
        incl_robot_future=False,
        incl_raster_map=False,
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=0,
        cache_location=TRAJDATA_CACHE_DIR,
        data_dirs={
            eval_dataset: ETH_UCY_RAW_DATA_DIR,
        },
        verbose=True
    )

    eval_dict = defaultdict(list)

    with torch.no_grad():
        # Base Model
        batch_eval_dataloader = get_dataloader(batch_eval_dataset, num_workers=16, batch_size=128)
        # batch_id = 0
        # for eval_batch in tqdm(batch_eval_dataloader, desc=f'Eval Base'):
        #     # base_trajectron.save_conflictive_scenarios(eval_batch, batch_id)
        #     # if batch_id == 7:
        #     print("BATCH: ", batch_id)
        #     print(base_trajectron.visualize_scenarios(eval_batch, batch_id))
        #     batch_id = batch_id + 1
        #     # if batch_id==5:
        #     #     break
        # # del eval_batch 
        rank = 0
        collect_metrics(batch_eval_dataloader, rank, base_trajectron)

def test_full(model_name):
    # %% 
    peds_datasets = ["eth", "hotel", "univ", "zara1", "zara2"]

    history_sec = 2.8
    prediction_sec = 4.8

    checkpoint = 10

    # train_scene = "zara1"
    # eval_scene = "eth"

    # base_model = glob.glob(f"kf_models/{train_scene}_*_base_*")[0]
    model_path = "experiments/pedestrians/kf_models/"
    base_model = model_path + model_name
    # eval_dataset = f"eupeds_{eval_scene}"
    # eval_data = f"{eval_dataset}-test_loo"

    base_trajectron, _ = load_model(base_model, device, epoch=checkpoint,
        custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                            "single_mode_multi_sample": False}
    )

    # Load training and evaluation environments and scenes
    attention_radius = defaultdict(lambda: 20.0) # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    batch_eval_dataset = UnifiedDataset(
        desired_data=["eupeds_eth-val", "eupeds_hotel-val", "eupeds_univ-val", "eupeds_zara1-val", "eupeds_zara2-val"],
        history_sec=(history_sec, history_sec),
        future_sec=(prediction_sec, prediction_sec),
        agent_interaction_distances=attention_radius,
        incl_robot_future=False,
        incl_raster_map=False,
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=0,
        cache_location=TRAJDATA_CACHE_DIR,
        data_dirs={
            "eupeds_eth": ETH_UCY_RAW_DATA_DIR,
            "eupeds_hotel": ETH_UCY_RAW_DATA_DIR,
            "eupeds_univ": ETH_UCY_RAW_DATA_DIR,
            "eupeds_zara1": ETH_UCY_RAW_DATA_DIR,
            "eupeds_zara2": ETH_UCY_RAW_DATA_DIR,
        },
        verbose=True
    )

    eval_dict = defaultdict(list)

    with torch.no_grad():
        # Base Model
        batch_eval_dataloader = get_dataloader(batch_eval_dataset, num_workers=16, batch_size=256)
        # batch_id = 0
        # for eval_batch in tqdm(batch_eval_dataloader, desc=f'Eval Base'):
        #     # base_trajectron.save_conflictive_scenarios(eval_batch, batch_id)
        #     # if batch_id == 7:
        #     print("BATCH: ", batch_id)
        #     print(base_trajectron.visualize_scenarios(eval_batch, batch_id))
        #     batch_id = batch_id + 1
        #     # if batch_id==5:
        #     #     break
        # # del eval_batch 
        rank = 0
        collect_metrics(batch_eval_dataloader, rank, base_trajectron)


# %%
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.set_device(0)
else:
    device = 'cpu'
print(torch.cuda.is_available())

# print("Full model")
# test_full("full_model_guidance_hcost_left-04_Oct_2023_19_00_01")
# print("ETH")
# test_dataset("eth", "eth_guidance_hcost_left_eth-14_Oct_2023_11_44_50")
# print("HOTEL")
# test_dataset("hotel", "eth_guidance_hcost_left_hotel-15_Oct_2023_10_15_48")
# print("UNIV")
# test_dataset("univ", "eth_guidance_hcost_left_univ-16_Oct_2023_08_24_40")
# print("ZARA1")
# test_dataset("zara1", "eth_guidance_hcost_left_zara1-16_Oct_2023_12_40_54")
# print("ZARA2")
# test_dataset("zara2", "eth_guidance_hcost_left_zara2-17_Oct_2023_10_58_31")

print("Full model")
test_full("full_model_guidance_hcost-24_Oct_2023_10_05_16")
print("ETH")
test_dataset("eth", "eth_guidance_hcost_eth-20_Oct_2023_15_52_42")
print("HOTEL")
test_dataset("hotel", "eth_guidance_hcost_hotel-21_Oct_2023_13_51_07")
print("UNIV")
test_dataset("univ", "eth_guidance_hcost_univ-22_Oct_2023_11_26_01")
print("ZARA1")
test_dataset("zara1", "eth_guidance_hcost_zara1-22_Oct_2023_15_36_59")
print("ZARA2")
test_dataset("zara2", "eth_guidance_hcost_zara2-23_Oct_2023_13_21_34")
