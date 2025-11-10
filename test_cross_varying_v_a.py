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
import time
import random


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
        print("Results: ", eval_final_results)
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
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.set_device(0)
else:
    device = 'cpu'
print(torch.cuda.is_available())

# %% 

history_sec = 2.8
prediction_sec = 4.8

checkpoint = 10


model_path = "experiments/pedestrians/kf_models/"
base_model = model_path + "full_model_guidance_hcost-02_Oct_2023_16_32_39"

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


def vary_v_a_exp():
    base_trajectron.n_topologies = 2
    max_steps = 8 + 13
    vel_robot = 1.25 * (1-np.arange(max_steps)/(1.1*21))
    num_model_based = 1
    vel_ped = 0.8
    # Initialize matrix
    pos_mat = np.zeros((2, 3, max_steps))
    for step_collision in [2,3,4,5,6,7,8,9,10]:
        # Fill matrix with robot and neighbour data
        pos_mat[0, :, :] = np.array([[(np.arange(max_steps)-7)*0.4*vel_robot, np.zeros(max_steps), np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        pos_mat[1, :, :] = np.array([[np.ones(max_steps)*0.4*step_collision*vel_robot, (np.arange(max_steps)-7-step_collision)*0.4*vel_ped, np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        neigh_hist_len = [8]
        angle_robot = -np.arctan2(pos_mat[0, 1, 7] - pos_mat[0, 1, 6], pos_mat[0, 0, 7] - pos_mat[0, 0, 6])
        sin_robot = np.sin(angle_robot)
        cos_robot = np.cos(angle_robot)
        tf_world_agent = np.array(
            [
                [cos_robot, -sin_robot, -pos_mat[0, 0, 7]*cos_robot + pos_mat[0, 1, 7]*sin_robot],
                [sin_robot, cos_robot, -pos_mat[0, 0, 7]*sin_robot - pos_mat[0, 1, 7]*cos_robot],
                [0.0, 0.0, 1.0],
            ]
        ) 
        # Every position in current robot's frame
        pos_mat = tf_world_agent@pos_mat
        # net input shape (batch_size, 1+n_obs, t, state_size)
        # state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta)
        net_input = torch.zeros((1, pos_mat.shape[0], pos_mat.shape[2], 8), device=device)
        net_input[:, :, :, :2] = torch.tensor(pos_mat[:, :2, :].transpose(0, 2, 1), device=device).unsqueeze(0)
        net_input[:, :, 1:, 2:4] = (net_input[:, :, 1:, :2] - net_input[:, :, :-1, :2])/0.4
        net_input[:, :, 1:, 4:6] = (net_input[:, :, 1:, 2:4] - net_input[:, :, :-1, 2:4])/0.4
        theta = torch.atan2(net_input[:, :, 1:, 3], net_input[:, :, 1:, 2])
        net_input[:, :, 1:, 6] = torch.sin(theta)
        net_input[:, :, 1:, 7] = torch.cos(theta)
        batch = AgentBatch(data_idx=None, scene_ts=None, dt=torch.tensor([0.4]), agent_name=None, agent_type=None, 
                            curr_agent_state=None, agent_hist=net_input[:, 0, :8, :], agent_hist_extent=None,
                            agent_hist_len=torch.tensor([8]), agent_fut=net_input[:, 0, 8:, :], 
                            agent_fut_extent=None, agent_fut_len=torch.tensor([13]), num_neigh=torch.tensor([1]), 
                            neigh_types=torch.tensor([[AgentType.PEDESTRIAN]]).repeat(1,net_input.shape[1]-1), 
                            neigh_hist=net_input[:, 1:, :8, :],
                            neigh_hist_extents=None, neigh_hist_len=torch.tensor([neigh_hist_len]), neigh_fut=net_input[:, 1:, 8:, :], neigh_fut_extents=None,
                            neigh_fut_len=torch.tensor([[13]]), robot_fut=None, robot_fut_len=None, maps=None, maps_resolution=None, 
                            vector_maps=None, rasters_from_world_tf=None, agents_from_world_tf=None, scene_ids=None,
                            history_pad_dir=None, extras={}, map_names=None)
        h_signatures = torch.tensor([[[0,1]]], device=device).unsqueeze(-1)
        selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_signatures)
        mean_h = []
        for _ in range(num_model_based):
            trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = base_trajectron.request_homotopic_info(batch, get_real=True)
            mean_h.append(h_tensor_combined[0,0,0,0])
        # selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_tensor_combined)

        # time.sleep(1)
        # print(torch.argmin(selected))
        # h_tensor_combined[0,0,0,0]
        plt.plot()
        print("In step_collision ", step_collision, ": Model-based: ", torch.tensor(mean_h).mean(), ". Learning based: ", torch.argmin(selected))
    print("Second experiment")
    for step_collision in [2,3,4,5,6,7,8,9,10]:
        # Fill matrix with robot and neighbour data
        pos_mat[0, :, :] = np.array([[(np.arange(max_steps)-7)*0.4*vel_robot, np.zeros(max_steps), np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        pos_mat[1, :, :] = np.array([[np.ones(max_steps)*0.4*6*vel_robot, (np.arange(max_steps)-7-step_collision)*0.4*vel_ped, np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        neigh_hist_len = [8]
        angle_robot = -np.arctan2(pos_mat[0, 1, 7] - pos_mat[0, 1, 6], pos_mat[0, 0, 7] - pos_mat[0, 0, 6])
        sin_robot = np.sin(angle_robot)
        cos_robot = np.cos(angle_robot)
        tf_world_agent = np.array(
            [
                [cos_robot, -sin_robot, -pos_mat[0, 0, 7]*cos_robot + pos_mat[0, 1, 7]*sin_robot],
                [sin_robot, cos_robot, -pos_mat[0, 0, 7]*sin_robot - pos_mat[0, 1, 7]*cos_robot],
                [0.0, 0.0, 1.0],
            ]
        ) 
        # Every position in current robot's frame
        pos_mat = tf_world_agent@pos_mat
        # net input shape (batch_size, 1+n_obs, t, state_size)
        # state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta)
        net_input = torch.zeros((1, pos_mat.shape[0], pos_mat.shape[2], 8), device=device)
        net_input[:, :, :, :2] = torch.tensor(pos_mat[:, :2, :].transpose(0, 2, 1), device=device).unsqueeze(0)
        net_input[:, :, 1:, 2:4] = (net_input[:, :, 1:, :2] - net_input[:, :, :-1, :2])/0.4
        net_input[:, :, 1:, 4:6] = (net_input[:, :, 1:, 2:4] - net_input[:, :, :-1, 2:4])/0.4
        theta = torch.atan2(net_input[:, :, 1:, 3], net_input[:, :, 1:, 2])
        net_input[:, :, 1:, 6] = torch.sin(theta)
        net_input[:, :, 1:, 7] = torch.cos(theta)
        batch = AgentBatch(data_idx=None, scene_ts=None, dt=torch.tensor([0.4]), agent_name=None, agent_type=None, 
                            curr_agent_state=None, agent_hist=net_input[:, 0, :8, :], agent_hist_extent=None,
                            agent_hist_len=torch.tensor([8]), agent_fut=net_input[:, 0, 8:, :], 
                            agent_fut_extent=None, agent_fut_len=torch.tensor([13]), num_neigh=torch.tensor([1]), 
                            neigh_types=torch.tensor([[AgentType.PEDESTRIAN]]).repeat(1,net_input.shape[1]-1), 
                            neigh_hist=net_input[:, 1:, :8, :],
                            neigh_hist_extents=None, neigh_hist_len=torch.tensor([neigh_hist_len]), neigh_fut=net_input[:, 1:, 8:, :], neigh_fut_extents=None,
                            neigh_fut_len=torch.tensor([[13]]), robot_fut=None, robot_fut_len=None, maps=None, maps_resolution=None, 
                            vector_maps=None, rasters_from_world_tf=None, agents_from_world_tf=None, scene_ids=None,
                            history_pad_dir=None, extras={}, map_names=None)
        h_signatures = torch.tensor([[[0,1]]], device=device).unsqueeze(-1)
        selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_signatures)
        mean_h = []
        for _ in range(num_model_based):
            trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = base_trajectron.request_homotopic_info(batch, get_real=True)
            mean_h.append(h_tensor_combined[0,0,0,0])
        # time.sleep(1)
        # print(torch.argmin(selected))
        # h_tensor_combined[0,0,0,0]
        print("In step_collision ", step_collision, ": Model-based: ", torch.tensor(mean_h).mean(), ". Learning based: ", torch.argmin(selected))
    print("Third experiment")
    for step_collision in [2,3,4,5,6,7,8,9,10]:
        # Fill matrix with robot and neighbour data
        pos_mat[0, :, :] = np.array([[(np.arange(max_steps)-7)*0.4*vel_robot, np.zeros(max_steps), np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        pos_mat[1, :, :] = np.array([[np.ones(max_steps)*0.4*step_collision*vel_robot, (np.arange(max_steps)-7-6)*0.4*vel_ped, np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
        neigh_hist_len = [8]
        angle_robot = -np.arctan2(pos_mat[0, 1, 7] - pos_mat[0, 1, 6], pos_mat[0, 0, 7] - pos_mat[0, 0, 6])
        sin_robot = np.sin(angle_robot)
        cos_robot = np.cos(angle_robot)
        tf_world_agent = np.array(
            [
                [cos_robot, -sin_robot, -pos_mat[0, 0, 7]*cos_robot + pos_mat[0, 1, 7]*sin_robot],
                [sin_robot, cos_robot, -pos_mat[0, 0, 7]*sin_robot - pos_mat[0, 1, 7]*cos_robot],
                [0.0, 0.0, 1.0],
            ]
        ) 
        # Every position in current robot's frame
        pos_mat = tf_world_agent@pos_mat
        # net input shape (batch_size, 1+n_obs, t, state_size)
        # state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta)
        net_input = torch.zeros((1, pos_mat.shape[0], pos_mat.shape[2], 8), device=device)
        net_input[:, :, :, :2] = torch.tensor(pos_mat[:, :2, :].transpose(0, 2, 1), device=device).unsqueeze(0)
        net_input[:, :, 1:, 2:4] = (net_input[:, :, 1:, :2] - net_input[:, :, :-1, :2])/0.4
        net_input[:, :, 1:, 4:6] = (net_input[:, :, 1:, 2:4] - net_input[:, :, :-1, 2:4])/0.4
        theta = torch.atan2(net_input[:, :, 1:, 3], net_input[:, :, 1:, 2])
        net_input[:, :, 1:, 6] = torch.sin(theta)
        net_input[:, :, 1:, 7] = torch.cos(theta)
        batch = AgentBatch(data_idx=None, scene_ts=None, dt=torch.tensor([0.4]), agent_name=None, agent_type=None, 
                            curr_agent_state=None, agent_hist=net_input[:, 0, :8, :], agent_hist_extent=None,
                            agent_hist_len=torch.tensor([8]), agent_fut=net_input[:, 0, 8:, :], 
                            agent_fut_extent=None, agent_fut_len=torch.tensor([13]), num_neigh=torch.tensor([1]), 
                            neigh_types=torch.tensor([[AgentType.PEDESTRIAN]]).repeat(1,net_input.shape[1]-1), 
                            neigh_hist=net_input[:, 1:, :8, :],
                            neigh_hist_extents=None, neigh_hist_len=torch.tensor([neigh_hist_len]), neigh_fut=net_input[:, 1:, 8:, :], neigh_fut_extents=None,
                            neigh_fut_len=torch.tensor([[13]]), robot_fut=None, robot_fut_len=None, maps=None, maps_resolution=None, 
                            vector_maps=None, rasters_from_world_tf=None, agents_from_world_tf=None, scene_ids=None,
                            history_pad_dir=None, extras={}, map_names=None)
        h_signatures = torch.tensor([[[0,1]]], device=device).unsqueeze(-1)
        selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_signatures)
        mean_h = []
        for _ in range(num_model_based):
            trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = base_trajectron.request_homotopic_info(batch, get_real=True)
            mean_h.append(h_tensor_combined[0,0,0,0])
        # time.sleep(1)
        # print(torch.argmin(selected))
        # h_tensor_combined[0,0,0,0]
        print("In step_collision ", step_collision, ": Model-based: ", torch.tensor(mean_h).mean(), ". Learning based: ", torch.argmin(selected))

def specific_scenario():
    base_trajectron.n_topologies = 2
    step_collision = 5
    vel_robot = 1.25
    max_steps = 8 + 13
    vel_ped = 0.8
    # Initialize matrix
    pos_mat = np.zeros((3, 3, max_steps))
    # Fill matrix with robot and neighbour data
    pos_mat[0, :, :] = np.array([[(np.arange(max_steps)-7)*0.4*vel_robot, np.zeros(max_steps), np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
    pos_mat[1, :, :] = np.array([[((np.arange(max_steps))*0.4*vel_ped)[::-1], np.ones(max_steps)*0.35, np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
    pos_mat[2, :, :] = np.array([[((np.arange(max_steps))*0.4*vel_ped)[::-1], -np.ones(max_steps)*0.35, np.ones(max_steps)]]) #+ (np.random.random((3,max_steps))-0.5)/5.
    neigh_hist_len = [8]
    angle_robot = -np.arctan2(pos_mat[0, 1, 7] - pos_mat[0, 1, 6], pos_mat[0, 0, 7] - pos_mat[0, 0, 6])
    sin_robot = np.sin(angle_robot)
    cos_robot = np.cos(angle_robot)
    tf_world_agent = np.array(
        [
            [cos_robot, -sin_robot, -pos_mat[0, 0, 7]*cos_robot + pos_mat[0, 1, 7]*sin_robot],
            [sin_robot, cos_robot, -pos_mat[0, 0, 7]*sin_robot - pos_mat[0, 1, 7]*cos_robot],
            [0.0, 0.0, 1.0],
        ]
    ) 
    # Every position in current robot's frame
    pos_mat = tf_world_agent@pos_mat
    # net input shape (batch_size, 1+n_obs, t, state_size)
    # state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta)
    net_input = torch.zeros((1, pos_mat.shape[0], pos_mat.shape[2], 8), device=device)
    net_input[:, :, :, :2] = torch.tensor(pos_mat[:, :2, :].transpose(0, 2, 1), device=device).unsqueeze(0)
    net_input[:, :, 1:, 2:4] = (net_input[:, :, 1:, :2] - net_input[:, :, :-1, :2])/0.4
    net_input[:, :, 1:, 4:6] = (net_input[:, :, 1:, 2:4] - net_input[:, :, :-1, 2:4])/0.4
    theta = torch.atan2(net_input[:, :, 1:, 3], net_input[:, :, 1:, 2])
    net_input[:, :, 1:, 6] = torch.sin(theta)
    net_input[:, :, 1:, 7] = torch.cos(theta)
    batch = AgentBatch(data_idx=None, scene_ts=None, dt=torch.tensor([0.4]), agent_name=None, agent_type=None, 
                        curr_agent_state=None, agent_hist=net_input[:, 0, :8, :], agent_hist_extent=None,
                        agent_hist_len=torch.tensor([8]), agent_fut=net_input[:, 0, 8:, :], 
                        agent_fut_extent=None, agent_fut_len=torch.tensor([13]), num_neigh=torch.tensor([1]), 
                        neigh_types=torch.tensor([[AgentType.PEDESTRIAN]]).repeat(1,net_input.shape[1]-1), 
                        neigh_hist=net_input[:, 1:, :8, :],
                        neigh_hist_extents=None, neigh_hist_len=torch.tensor([neigh_hist_len]), neigh_fut=net_input[:, 1:, 8:, :], neigh_fut_extents=None,
                        neigh_fut_len=torch.tensor([[13]]), robot_fut=None, robot_fut_len=None, maps=None, maps_resolution=None, 
                        vector_maps=None, rasters_from_world_tf=None, agents_from_world_tf=None, scene_ids=None,
                        history_pad_dir=None, extras={}, map_names=None)
    h_signatures = torch.tensor([[[0,1]]], device=device).unsqueeze(-1)
    trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = base_trajectron.request_homotopic_info(batch, get_real=True)
    # selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_signatures)
    plt.figure()
    plt.show()
    # mean_h = []
    # for _ in range(num_model_based):
    #     trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = base_trajectron.request_homotopic_info(batch, get_real=True)
    #     mean_h.append(h_tensor_combined[0,0,0,0])
    # time.sleep(1)
    # print(torch.argmin(selected))
    # h_tensor_combined[0,0,0,0]
    # print("In step_collision ", step_collision, ": Model-based: ", torch.tensor(mean_h).mean(), ". Learning based: ", torch.argmin(selected))

specific_scenario()
exit()
# %%
# metrics_list = ['ml_ade', 'ml_fde', 'min_ade_5', 'min_fde_5', 'min_ade_10', 'min_fde_10', 'nll_mean', 'nll_final']

# %% [markdown]
# ### Qualitative Plots

# %%
# eval_dict = defaultdict(list)

eval_scene = "zara1"

eval_dataset = f"eupeds_{eval_scene}"
eval_data = f"{eval_dataset}-test_loo"

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

with torch.no_grad():
    # Base Model
    batch_eval_dataloader = get_dataloader(batch_eval_dataset, num_workers=16, batch_size=1)
    batch_id = 0
    for eval_batch in tqdm(batch_eval_dataloader, desc=f'Eval Base'):
        # base_trajectron.save_conflictive_scenarios(eval_batch, batch_id)
        # if batch_id == 7:
        # if batch_id < 44:
        #     batch_id = batch_id + 1
        #     continue
        print("BATCH: ", batch_id)
        # print(eval_batch)
        # h_signatures = torch.tensor([[[0,0],[0,1],[1,0],[1,1]]], device=device)
        # h_signatures = torch.tensor([[[[0],
        #                                 [0],
        #                                 [0],
        #                                 [1]]],


        #                                 [[[1],
        #                                 [0],
        #                                 [1],
        #                                 [1]]]], device=device)
        # base_trajectron.estimate_guidance_costs(batch=eval_batch, h_tensor=h_signatures)
        # exit(0)
        # time.sleep(1)
        print(base_trajectron.predict_and_evaluate_batch(eval_batch, batch_id))
        batch_id = batch_id + 1
        # if batch_id==5:
        #     break
    # # del eval_batch 
    # rank = 0
    # collect_metrics(batch_eval_dataloader, rank, base_trajectron)