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

from guidance_planner.srv import select_guidance, select_guidanceRequest, select_guidanceResponse
from guidance_planner.msg import RefAvoidanceMSG, ObstacleMSG
import rospy

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %%
# Change this to suit your computing environment and folder structure!

TRAJDATA_CACHE_DIR: Final[str] = "/home/diego/.unified_data_cache"
# ETH_UCY_RAW_DATA_DIR: Final[str] = "/home/diego/datasets/eth_ucy_peds"

# %%
# AXHLINE_COLORS = {
#     "Base": "#DD9787",
#     "K0": "#A6C48A",
#     "Oracle": "#BCB6FF"
# }

# SEABORN_PALETTE = {
#     "Finetune": "#AA7C85",
#     "K0+Finetune": "#2D93AD",
#     "Ours": "#67934D",
    
#     "Base": "#DD9787",
#     "K0": "#A6C48A",
#     "Oracle": "#BCB6FF"
# }

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

# # %%
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.set_device(0)
else:
    device = 'cpu'
print(torch.cuda.is_available())

# # %% 
checkpoint = 10

model_path = "experiments/pedestrians/kf_models/"
base_model = model_path + "full_model_guidance_hcost-24_Oct_2023_10_05_16"

base_trajectron, _ = load_model(base_model, device, epoch=checkpoint,
    custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                        "single_mode_multi_sample": False}
)


def handle_select_guidance(req: select_guidanceRequest):
    # print("---------------------------------------------------")
    # pos_mat is matrix with size (1 + n_obs, 3, n_steps)
    # In first dimension, the 1 + refers to the robot
    # Second dimension is (x, y, theta)
    base_trajectron.n_topologies = len(req.h_signatures)
    max_steps = 8
    # Initialize matrix
    pos_mat = np.zeros((len(req.previous_obstacles) + 1, 3, max_steps))
    # Fill matrix with robot and neighbour data
    pos_mat[0, :, -len(req.robot_trajectory.x):] = np.array([[req.robot_trajectory.x, req.robot_trajectory.y, np.ones_like(req.robot_trajectory.x)]])
    for obs_i in range(len(req.previous_obstacles)):
        pos_mat[obs_i + 1, :, -len(req.previous_obstacles[obs_i].pos_x):] = np.array([req.previous_obstacles[obs_i].pos_x, req.previous_obstacles[obs_i].pos_y, np.ones_like(req.previous_obstacles[obs_i].pos_x)])
    neigh_hist_len = torch.tensor(np.array([[len(o.pos_x) for o in req.previous_obstacles]]))
    angle_robot = -np.arctan2(pos_mat[0, 1, -1] - pos_mat[0, 1, -2], pos_mat[0, 0, -1] - pos_mat[0, 0, -2])
    sin_robot = np.sin(angle_robot)
    cos_robot = np.cos(angle_robot)
    tf_world_agent = np.array(
        [
            [cos_robot, -sin_robot, -pos_mat[0, 0, -1]*cos_robot + pos_mat[0, 1, -1]*sin_robot],
            [sin_robot, cos_robot, -pos_mat[0, 0, -1]*sin_robot - pos_mat[0, 1, -1]*cos_robot],
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
    batch = AgentBatch(data_idx=None, scene_ts=None, dt=None, agent_name=None, agent_type=None, 
                       curr_agent_state=None, agent_hist=net_input[:, 0, :, :], agent_hist_extent=None,
                       agent_hist_len=torch.tensor([len(req.robot_trajectory.x)]), agent_fut=None, 
                       agent_fut_extent=None, agent_fut_len=None, num_neigh=torch.tensor([net_input.shape[1]-1]), 
                       neigh_types=torch.tensor([[AgentType.PEDESTRIAN]]).repeat(1,net_input.shape[1]-1), 
                       neigh_hist=net_input[:, 1:, :, :],
                       neigh_hist_extents=None, neigh_hist_len=neigh_hist_len, neigh_fut=None, neigh_fut_extents=None,
                       neigh_fut_len=None, robot_fut=None, robot_fut_len=None, maps=None, maps_resolution=None, 
                       vector_maps=None, rasters_from_world_tf=None, agents_from_world_tf=None, scene_ids=None,
                       history_pad_dir=None, extras={}, map_names=None)
    h_signatures = torch.tensor([h.ref_avoidance for h in req.h_signatures], device=device).transpose(1,0).unsqueeze(1).unsqueeze(-1)
    selected = base_trajectron.estimate_guidance_costs(batch=batch, h_tensor=h_signatures)
    # plt.plot(pos_mat[0, 0, :], pos_mat[0, 1, :], 'bo')
    # plt.plot(pos_mat[1, 0, :], pos_mat[1, 1, :], 'ro')
    # plt.plot(pos_mat[2, 0, :], pos_mat[2, 1, :], 'ro')
    # plt.show()
    return select_guidanceResponse(selected.cpu().detach().numpy())

with torch.no_grad():
    rospy.init_node('network_server')
    s = rospy.Service('/select_guidance', select_guidance, handle_select_guidance)
    print("Ready to estimate cost.")
    rospy.spin()

    # # Simple program to test
    # request = select_guidanceRequest()
    # h_sig = RightAvoidanceMSG()
    # h_sig.ref_avoidance = [1.0, 1.0]
    # request.h_signatures.append(h_sig)
    # h_sig2 = RightAvoidanceMSG()
    # h_sig2.ref_avoidance = [0.0, 0.0]
    # request.h_signatures.append(h_sig2)
    # h_sig3 = RightAvoidanceMSG()
    # h_sig3.ref_avoidance = [1.0, 0.0]
    # request.h_signatures.append(h_sig2)
    # h_sig4 = RightAvoidanceMSG()
    # h_sig4.ref_avoidance = [0.0, 1.0]
    # request.h_signatures.append(h_sig4)
    # request.robot_trajectory.x = np.arange(8)*0.5
    # request.robot_trajectory.y = np.arange(8)*0.0
    # obs_msg = ObstacleMSG()
    # obs_msg.id = 0
    # obs_msg.radius = 0.2
    # obs_msg.pos_x = np.arange(8)*0.5
    # obs_msg.pos_y = np.arange(8)*0.0 + 0.7
    # request.previous_obstacles.append(obs_msg)
    # obs_msg2 = ObstacleMSG()
    # obs_msg2.id = 1
    # obs_msg2.radius = 0.2
    # obs_msg2.pos_x = (16-np.arange(8))*0.5
    # obs_msg2.pos_y = np.arange(8)*0.0 - 0.7
    # request.previous_obstacles.append(obs_msg2)
    # handle_select_guidance(request)
