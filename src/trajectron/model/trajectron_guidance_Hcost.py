# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from trajdata import AgentBatch, AgentType

from trajectron.model.mgcvae import MultimodalGenerativeCVAE
from trajectron.model.model_utils import *
from trajectron.model.trajectron import Trajectron

import rospy
from guidance_planner.srv import guidancesRequest, guidances, guidances_Hcost, guidances_HcostRequest
from guidance_planner.msg import TrajectoryMSG, ObstacleMSG

from trajectron.model.plotting import plot_scenario_dynamic

import matplotlib.pyplot as plt
import struct
import copy

class EncoderWithH(MultimodalGenerativeCVAE):
    def __init__(self, original_model, n_topologies, model_registrar):
        super().__init__(original_model.node_type_obj, original_model.model_registrar, original_model.hyperparams, original_model.device, original_model.edge_types, original_model.log_writer)
        self.hyperparams = original_model.hyperparams
        self.node_type_obj = original_model.node_type_obj
        self.node_type = original_model.node_type
        self.model_registrar = model_registrar
        self.log_writer = original_model.log_writer
        self.device = original_model.device
        self.edge_types = original_model.edge_types
        self.curr_iter = original_model.curr_iter
        self.clear_submodules()
        for key in original_model.node_modules.keys():
            if (key == self.node_type + "/map_encoder" or key == self.node_type + "/node_history_encoder" or
            "edge_encoder" in key or key == self.node_type + "/edge_influence_encoder"):
                self.add_submodule(key, original_model.node_modules[key])
        self.state = original_model.state
        self.pred_state = original_model.pred_state
        self.state_length = original_model.state_length
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = original_model.robot_state_length
        self.pred_state_length = original_model.pred_state_length
        self.dynamic = original_model.dynamic
        self.n_topologies = n_topologies
        self.add_submodule(self.node_type + "/reduce_edge_dim", nn.Linear(in_features=33, out_features=32, device=self.device))
        self.add_submodule(self.node_type + "/fc1", nn.Linear(64, 256, device=self.device))
        self.add_submodule(self.node_type + "/fc2", nn.Linear(256, 128, device=self.device))
        self.add_submodule(self.node_type + "/fc3", nn.Linear(128, 1, device=self.device)) 

    def get_cost_from_encoders(self, x: torch.Tensor, mode: ModeKeys):
        pred = F.dropout(
                F.relu((self.node_modules[self.node_type + "/fc1"](x))),
                p=0.1,
                training=(mode == ModeKeys.TRAIN),
            ) 
        pred = F.dropout(
                F.relu((self.node_modules[self.node_type + "/fc2"](pred))),
                p=0.1,
                training=(mode == ModeKeys.TRAIN),
            )   
        out = (self.node_modules[self.node_type + "/fc3"](pred)).squeeze(-1)
        return out
    
    def obtain_encoded_tensors(
        self, mode: ModeKeys, batch: AgentBatch, extra_node_info: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        enc, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        batch_size = batch.agent_hist.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history_st = batch.agent_hist
        node_history_st_len = batch.agent_hist_len
        node_present_state_st = node_history_st[
            torch.arange(node_history_st.shape[0]), node_history_st_len - 1
        ]

        initial_dynamics["pos"] = node_present_state_st[:, 0:2]
        initial_dynamics["vel"] = node_present_state_st[:, 2:4]

        self.dynamic.set_initial_condition(initial_dynamics)

        if self.hyperparams["incl_robot_node"]:
            robot = batch.robot_fut
            robot_lens = batch.robot_fut_len
            x_r_t, y_r = robot[:, 0], robot[:, 1:]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(
            mode, node_history_st, node_history_st_len
        )
        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = batch.agent_fut[..., :2]
            y_lens = batch.agent_fut_len

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams["edge_encoding"]:
            if batch.num_neigh.max() == 0:
                total_edge_influence = torch.zeros_like(node_history_encoded).unsqueeze(1).repeat((1,4,1))
            else:
                # Encode edges
                encoded_edges = self.encode_edge(
                    mode,
                    node_history_st,
                    node_history_st_len,
                    batch.neigh_hist,
                    batch.neigh_hist_len,
                    batch.neigh_types,
                    batch.num_neigh,
                )
                total_edge_influence = torch.empty(batch_size, 0, encoded_edges.shape[-1], device=self.device)
                for homology in range(self.n_topologies):
                    encoded_edges_h = F.relu(self.node_modules[self.node_type + "/reduce_edge_dim"](
                        torch.concat((encoded_edges, extra_node_info[:len(encoded_edges), :, homology, :]), -1)))
                    #####################
                    # Encode Node Edges #
                    #####################

                    total_edge_influence_i, attn_weights = self.encode_total_edge_influence(
                        mode,
                        encoded_edges_h,
                        batch.num_neigh,
                        node_history_encoded,
                        node_history_st_len,
                        batch_size,
                    )
                    total_edge_influence = torch.cat((total_edge_influence, total_edge_influence_i.unsqueeze(1)), 1)

        ################
        # Map Encoding #
        ################
        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if (
                self.hyperparams["log_maps"]
                and self.log_writer
                and (self.curr_iter + 1) % 500 == 0
            ):
                image = wandb.Image(batch.maps[0], caption=f"Batch Map 0")
                self.log_writer.log(
                    {f"{self.node_type}/maps": image}, step=self.curr_iter, commit=False
                )

            encoded_map = self.node_modules[self.node_type + "/map_encoder"](
                batch.maps * 2.0 - 1.0, (mode == ModeKeys.TRAIN)
            )
            do = self.hyperparams["map_encoder"][self.node_type]["dropout"]
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        enc_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams["edge_encoding"]:
            enc_concat_list.append(total_edge_influence)  # [bs/nbs, enc_rnn_dim]

        # Every node has a history encoder.
        node_history_encoded = node_history_encoded.unsqueeze(1).repeat(1, self.n_topologies, 1)
        enc_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams["incl_robot_node"]:
            robot_future_encoder = self.encode_robot_future(
                mode, x_r_t, y_r, robot_lens
            )
            enc_concat_list.append(robot_future_encoder)

        if (
            self.hyperparams["map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if self.log_writer:
                self.log_writer.log(
                    {
                        f"{self.node_type}/encoded_map_max": torch.max(
                            torch.abs(encoded_map)
                        ).item()
                    },
                    step=self.curr_iter,
                    commit=False,
                )
            enc_concat_list.append(
                encoded_map.unsqueeze(1).expand((-1, node_history_encoded.shape[1], -1))
                if self.hyperparams["adaptive"]
                else encoded_map
            )
        enc = torch.cat(enc_concat_list, dim=-1)


        return enc

class TrajectronGuidanceHCost(Trajectron):
    def __init__(self, original_model, model_registrar):
        super().__init__(original_model.model_registrar, original_model.hyperparams, original_model.log_writer, original_model.device)
        self.hyperparams = original_model.hyperparams
        self.log_writer = original_model.log_writer
        self.device = original_model.device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = original_model.node_models_dict
        self.nodes = original_model.nodes

        self.state = original_model.state
        self.state_length = original_model.state_length
        self.pred_state = original_model.pred_state

        self.loss_function = nn.MSELoss()
        self.dt = 0.4
        self.person_radius = 0.45
        self.n_topologies = 4
        for key in self.node_models_dict.keys():
            if isinstance(self.node_models_dict[key], MultimodalGenerativeCVAE):
                self.node_models_dict[key] = EncoderWithH(self.node_models_dict[key], self.n_topologies, model_registrar)
        self.topologies_dict = {}

    def get_encoded_state(self, batch: AgentBatch):
        batch.to(self.device)
        node_type: AgentType
        enc_dict = {}
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)
        
            mode = ModeKeys.PREDICT
            enc = model.obtain_encoded_tensors(mode, agent_type_batch)     
            enc_np = enc.cpu().detach().numpy()
            for i, agent_name in enumerate(agent_type_batch.agent_name):
                enc_dict[f"{str(node_type)}/{agent_name}/{str(i)}"] = enc_np[
                    i, :
            ]
        return enc_dict
    
    def predict(
        self,
        batch: AgentBatch,
        update_mode: UpdateMode = UpdateMode.NO_UPDATE,
        num_samples=1,
        prediction_horizon=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=True,
    ):
        batch.to(self.device)
        node_type: AgentType
        pred_dict = {}
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)
        
            mode = ModeKeys.PREDICT
            trajs_tensor_combined, ground_truth_batch, success, num_h, h_tensor_combined = self.request_homotopic_info(batch, get_real=True)
            enc = model.obtain_encoded_tensors(mode, agent_type_batch, h_tensor_combined)  
            pred = model.get_cost_from_encoders(enc, mode)
            success = np.array(success)
            num_h = np.array(num_h)
            pred = pred[success]
            num_h = num_h[success]
            pred_np = pred.cpu().detach().numpy()
            for i, agent_name in enumerate(agent_type_batch.agent_name):
                pred_dict[f"{str(node_type)}/{agent_name}/{str(i)}"] = pred_np[
                    i, :
            ]
        return pred_dict
    
    def request_homotopic_info(self, batch, get_real=False, batch_range=None, check_stored=False):
        # For now, we assume that the obstacles keep their current velocities
        # batch.neigh_hist: [bs, obstacles, times_tracked, state_size]
        #                   batch.neigh_hist[:, :, -1, :] -> Current state
        #                   state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta) Relative to the robot's current frame
        # batch.neigh_fut: [bs, obstacles, times_fut, state_size]
        #                   batch.neigh_hist[:, :, 0, :] -> Next state
        #                   state: x,y,xd,yd,xdd,ydd,sin(theta),cos(theta) Relative to the robot's current frame

        batch.to(self.device)
        ground_truth = None
        # trajs_tensor_combined = torch.empty((0, self.n_topologies, batch.neigh_fut.shape[2]+1, 2), device=self.device)
        h_tensor_combined = torch.empty((batch.neigh_fut.shape[1], 0, self.n_topologies, 1), device=self.device)        
        success=[]
        ground_truth_batch = torch.empty((0, self.n_topologies), device=self.device)
        num_h = []
        if batch_range == None:
            batch_range = range(batch.neigh_fut.shape[0])
        for batch_i in batch_range:
            obstacles_vec = []
            for obstacle_i in range(batch.neigh_fut.shape[1]):
                if not batch.neigh_fut[batch_i, obstacle_i,0,0].isnan():
                    # pos_x = batch.neigh_fut[batch_i, obstacle_i, 0, 0] + torch.arange(-1, batch.agent_fut.shape[1], 
                    #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 2] * self.dt
                    # pos_y = batch.neigh_fut[batch_i, obstacle_i, 0, 1] + torch.arange(-1, batch.agent_fut.shape[1], 
                    #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 3] * self.dt
                    pos_x = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 0].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 0]])
                    pos_y = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 1].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 1]])
                    for nan_idx in pos_x.isnan().nonzero():
                        if nan_idx[0] == 0:
                            pos_x[nan_idx] = pos_x[1] - batch.neigh_fut[batch_i, obstacle_i, 0, 2] * self.dt
                            pos_y[nan_idx] = pos_y[1] - batch.neigh_fut[batch_i, obstacle_i, 0, 2] * self.dt
                        else:
                            # Neither pos[0] nor pos[1] can be nan at this point, so pos[nan_idx-1] and pos[nan_idx-2] are not nan
                            pos_x[nan_idx] = 2*pos_x[nan_idx-1] - pos_x[nan_idx-2]
                            pos_y[nan_idx] = 2*pos_y[nan_idx-1] - pos_y[nan_idx-2]
                    obstacles_vec.append(ObstacleMSG(id=obstacle_i, pos_x=pos_x, pos_y=pos_y, radius=self.person_radius))   
            # goals within a radius
            goal_radius = 0.5
            goal_x = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,0]
            goals_x = [goal_x, goal_x + goal_radius, goal_x + goal_radius, goal_x - goal_radius, goal_x - goal_radius]
            goal_y = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,1]
            goals_y = [goal_y, goal_y + goal_radius, goal_y - goal_radius, goal_y + goal_radius, goal_y - goal_radius]      
            if get_real:
                topologies = self.get_topologies_Hcost(obstacles=obstacles_vec, x=batch.agent_hist[batch_i,-1,0], y=batch.agent_hist[batch_i,-1,1], 
                                             orientation=math.atan2(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]), 
                                             v=torch.norm(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]), 
                                             goals_x=goals_x, goals_y=goals_y, static_x=[], static_y=[], static_n=[], n_trajectories=4, 
                                             truth=TrajectoryMSG(x=torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,0]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0]]),
                                                                 y=torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,1]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1]])),
                                             data_idx=batch.data_idx, check_stored=check_stored)

                success.append(topologies[0])
            else:
                topologies = self.get_topologies(obstacles=obstacles_vec, x=batch.agent_hist[batch_i,-1,0], y=batch.agent_hist[batch_i,-1,1], 
                                             orientation=math.atan2(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]), 
                                             v=torch.norm(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]), 
                                             goals_x=goals_x, goals_y=goals_y, static_x=[], static_y=[], static_n=[], n_trajectories=4)
                success.append(topologies[0] and (np.asarray(topologies[2])<0.25).any())
            num_h.append(len(topologies[1]))
            if get_real:
                ground_truth = torch.zeros((1, self.n_topologies), device=self.device) - 1.0
                if topologies[0]:
                    ground_truth[:, :len(topologies[2])] = torch.tensor(np.asarray(topologies[2]), device=self.device)
                ground_truth_batch = torch.cat((ground_truth_batch, ground_truth), 0)
            # if topologies[0]:
            #     trajs_tensor = torch.tensor([[t.x, t.y] for t in topologies[1]], device=self.device).permute(0,2,1)
            #     trajs_tensor = trajs_tensor.repeat(self.n_topologies, 1, 1)[:self.n_topologies, :]
            # else:
            #     trajs_tensor = torch.zeros((4,batch.neigh_fut.shape[2]+1,2), device=self.device)
            # if trajs_tensor.shape[1] != batch.neigh_fut.shape[2]+1:
            #     # some trajectories may be shorter
            #     trajs_tensor = torch.hstack([trajs_tensor, trajs_tensor[:,-1,:].unsqueeze(1).repeat([1,batch.neigh_fut.shape[2]+1 - trajs_tensor.shape[1],1])])
            # trajs_tensor_combined = torch.cat((trajs_tensor_combined, trajs_tensor.unsqueeze(0)), 0)
            h_tensor = torch.zeros((1, self.n_topologies, batch.neigh_fut.shape[1]), device=self.device)
            raw_h = torch.tensor(np.array([np.asarray(t.ref_avoidance) for t in topologies[-1]]))
            if len(raw_h) > 0:
                h_tensor[0, :raw_h.shape[0], :raw_h.shape[1]] = raw_h
            h_tensor = h_tensor.permute(2,0,1).unsqueeze(-1)
            h_tensor_combined = torch.cat((h_tensor_combined, h_tensor), 1)
        # return trajs_tensor_combined, ground_truth_batch, success, num_h, h_tensor_combined
        return [], ground_truth_batch, success, num_h, h_tensor_combined
            # Plot for debug or visualize
            #-------------------------------------------------------------------------------------------------------
            # self.plot_scenario(topologies, obstacles_vec, batch, batch_i)
            # self.write_scenario_file(batch_i, obstacles_vec, batch, goals_x, goals_y)
            #------------------------------------------------------------------------------------------------------
    
    
    def plot_scenario(self, topologies, obstacles_vec, batch, batch_i):
            # Plot for debug or visualize
            # -------------------------------------------------------------------------------------------------------
            # if len(topologies[1]) > 1:
            plt.figure()
            ax = plt.axes(projection='3d')
            for obstacle in obstacles_vec:
                plt.plot(obstacle.pos_x.cpu().numpy(), obstacle.pos_y.cpu().numpy(), np.arange(len(obstacle.pos_x.cpu().numpy())), 'b-')
                plt.plot(obstacle.pos_x[0].cpu().numpy(), obstacle.pos_y[0].cpu().numpy(), 0, 'bx')
                # plt.plot(obstacle.pos_x[-1].cpu().numpy(), obstacle.pos_y[-1].cpu().numpy(), 'bx')

            plt.plot(batch.agent_hist[batch_i,-1,0].cpu().numpy(), batch.agent_hist[batch_i,-1,1].cpu().numpy(), 0, 'rx')
            # plt.plot(batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,0].cpu().numpy(), batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,1].cpu().numpy(), 'rx')
            plt.plot(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0].cpu().numpy(), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1].cpu().numpy(), np.arange(len(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0].cpu().numpy())), 'g-')
            minx = np.inf
            maxx = -np.inf
            miny = np.inf
            maxy = -np.inf
            for topology in topologies[1]:
                plt.plot(topology.x, topology.y, np.arange(len(topology.x)), 'r-')
                minx = min(topology.x)
                miny = min(topology.y)
                maxx = max(topology.x)
                maxy = max(topology.y)
            # plt.xlim(min(minx, min(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0].cpu().numpy())), max(maxx, max(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0].cpu().numpy())))
            # plt.ylim(min(miny, min(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1].cpu().numpy())), max(maxy, max(batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1].cpu().numpy())))
            plt.show()       
    
    def write_scenario_file(self, batch_i, obstacles_vec, batch, goals_x, goals_y):
        with open("scenario_" + str(batch_i) + ".bin", "wb") as f:
            # Obstacles vec
            data = struct.pack("i", len(obstacles_vec))
            f.write(data)
            for obs in obstacles_vec:
                data = struct.pack("i", len(obs.pos_x))
                f.write(data)
                for x_obs in obs.pos_x:
                    data = struct.pack("d", x_obs)
                    f.write(data)
                for y_obs in obs.pos_y:
                    data = struct.pack("d", y_obs)
                    f.write(data)
            # x
            data = struct.pack("d", batch.agent_hist[batch_i,-1,0])
            f.write(data)
            # y
            data = struct.pack("d", batch.agent_hist[batch_i,-1,1])
            f.write(data)
            # orientation
            data = struct.pack("d", math.atan2(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]))
            f.write(data)
            # v
            data = struct.pack("d", torch.norm(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]))
            f.write(data)
            # goals
            data = struct.pack("i", len(goals_x))
            f.write(data)
            for goal_x in goals_x:
                data = struct.pack("d", goal_x)
                f.write(data)
            for goal_y in goals_y:
                data = struct.pack("d", goal_y)
                f.write(data)
            # human truth
            x_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,0]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0]])
            y_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,1]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1]])
            data = struct.pack("i", len(x_traj))
            f.write(data)
            for x_t in x_traj:
                data = struct.pack("d", x_t)
                f.write(data)
            for y_t in y_traj:
                data = struct.pack("d", y_t)
                f.write(data)
    
    def write_scenario_file_truth(self, batch_i, obstacles_vec, batch, goals_x, goals_y, obstacles_previous,
                                  human_previous, baseline_traj, ours_traj, dir, batch_id):
        with open(dir + "/scenario_" + str(batch_id) + "_" + str(batch_i) + ".bin", "wb") as f:
            # Obstacles vec
            data = struct.pack("i", len(obstacles_vec))
            f.write(data)
            for obs in obstacles_previous:
                data = struct.pack("i", len(obs.pos_x))
                f.write(data)
                for x_obs in obs.pos_x:
                    data = struct.pack("d", x_obs)
                    f.write(data)
                for y_obs in obs.pos_y:
                    data = struct.pack("d", y_obs)
                    f.write(data)
            for obs in obstacles_vec:
                data = struct.pack("i", len(obs.pos_x))
                f.write(data)
                for x_obs in obs.pos_x:
                    data = struct.pack("d", x_obs)
                    f.write(data)
                for y_obs in obs.pos_y:
                    data = struct.pack("d", y_obs)
                    f.write(data)
            # x
            data = struct.pack("d", batch.agent_hist[batch_i,-1,0])
            f.write(data)
            # y
            data = struct.pack("d", batch.agent_hist[batch_i,-1,1])
            f.write(data)
            # orientation
            data = struct.pack("d", math.atan2(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]))
            f.write(data)
            # v
            data = struct.pack("d", torch.norm(batch.agent_hist[batch_i,-1,3], batch.agent_hist[batch_i,-1,2]))
            f.write(data)
            # goals
            data = struct.pack("i", len(goals_x))
            f.write(data)
            for goal_x in goals_x:
                data = struct.pack("d", goal_x)
                f.write(data)
            for goal_y in goals_y:
                data = struct.pack("d", goal_y)
                f.write(data)
            # human truth
            x_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,0]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0]])
            y_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,1]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1]])
            data = struct.pack("i", len(x_traj))
            f.write(data)
            for x_t in x_traj:
                data = struct.pack("d", x_t)
                f.write(data)
            for y_t in y_traj:
                data = struct.pack("d", y_t)
                f.write(data)
            data = struct.pack("i", len(baseline_traj.pos_x))
            f.write(data)
            for x_t in baseline_traj.pos_x:
                data = struct.pack("d", x_t)
                f.write(data)
            for y_t in baseline_traj.pos_y:
                data = struct.pack("d", y_t)
                f.write(data)
            for x_t in ours_traj.pos_x:
                data = struct.pack("d", x_t)
                f.write(data)
            for y_t in ours_traj.pos_y:
                data = struct.pack("d", y_t)
                f.write(data)
            data = struct.pack("i", len(human_previous.pos_x))
            f.write(data)
            for x_t in human_previous.pos_x:
                data = struct.pack("d", x_t)
                f.write(data)
            for y_t in human_previous.pos_y:
                data = struct.pack("d", y_t)
                f.write(data)
    
    def train_loss(
        self, batch: AgentBatch, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR
    ):
        batch.to(self.device)

        # Run forward pass
        losses: List[torch.Tensor] = list()
        node_type: AgentType
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]
            agent_type_batch = batch.for_agent_type(node_type)
            losses.append(self.MSE_train_loss(agent_type_batch, model, update_mode))

        return sum(losses)

    def MSE_train_loss(
        self, batch: AgentBatch, model: EncoderWithH, update_mode: UpdateMode = UpdateMode.BATCH_FROM_PRIOR,
    ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN
        trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = self.request_homotopic_info(batch, get_real=True, check_stored=True)
        enc = model.obtain_encoded_tensors(mode, batch, h_tensor_combined)  
        pred = model.get_cost_from_encoders(enc, mode)
        success = np.array(success)
        num_h = np.array(num_h)
        pred = pred[success]
        ground_truth = ground_truth[success]
        num_h = num_h[success]

        if self.hyperparams["adaptive"]:
            pos_hist: torch.Tensor = batch.agent_hist[..., :2]
        else:
            pos_hist: torch.Tensor = batch.agent_hist[
                torch.arange(batch.agent_hist.shape[0]), batch.agent_hist_len - 1
            ]
        loss = self.loss_function(pred[ground_truth!=-1], ground_truth[ground_truth!=-1])

        return loss
    
    def estimate_guidance_costs(self, batch: AgentBatch, h_tensor: torch.Tensor):
        batch.to(self.device)
        model: EncoderWithH = self.node_models_dict['PEDESTRIAN']
        model.n_topologies = self.n_topologies
        mode = ModeKeys.PREDICT
        enc = model.obtain_encoded_tensors(mode, batch=batch, extra_node_info=h_tensor)  
        pred = model.get_cost_from_encoders(enc, mode)
        return pred[0]

    def predict_and_evaluate_batch(
        self,
        batch: AgentBatch,
        update_mode: UpdateMode = UpdateMode.NO_UPDATE,
        output_for_pd: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[AgentType, Dict[str, torch.Tensor]]]:
        """Predicts from a batch and then evaluates the output, returning the batched errors."""
        batch.to(self.device)

        # Run forward pass
        results: Dict[AgentType, Dict[str, torch.Tensor]] = dict()

        node_type: AgentType
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            mode = ModeKeys.PREDICT
            trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = self.request_homotopic_info(batch, get_real=True)
            enc = model.obtain_encoded_tensors(mode, agent_type_batch, h_tensor_combined)  
            pred = model.get_cost_from_encoders(enc, mode)
            success = np.array(success)
            num_h = np.array(num_h)
            pred = pred[success]
            ground_truth = ground_truth[success]
            num_h = num_h[success]
            y_acc = []
            base_acc = []
            for i_acc in range(len(pred)):
                if num_h[i_acc] > 1:
                    y_acc.append(torch.argmin(pred[i_acc, :num_h[i_acc]]) == torch.argmin(ground_truth[i_acc, :num_h[i_acc]]))
                    base_acc.append(torch.argmin(ground_truth[i_acc, :num_h[i_acc]])==0)
            if len(y_acc) > 0:
                acc = sum(y_acc)/len(y_acc)
                baseline_accuracy = sum(base_acc)/len(y_acc)
                mean_hom_scenario = sum(num_h[num_h>1])/sum(num_h>1)
                loss = self.loss_function(pred[ground_truth!=-1], ground_truth[ground_truth!=-1])
            else:
                acc = float("nan")
                baseline_accuracy = float("nan")
                mean_hom_scenario = float("nan")
                loss = float("nan")
            # batch_eval: Dict[
            #     str, torch.Tensor
            # ] = evaluation.compute_batch_statistics_pt(
            #     agent_type_batch.agent_fut[..., :2],
            #     prediction_output_dict=predictions,
            #     y_dists=y_dists,
            # )
            results[node_type] = {"accuracy": torch.tensor([acc]), "scenarios_homologies": torch.tensor([sum(num_h>1)]), 
                                  "failed_scenarios": torch.tensor([sum(np.logical_not(success))]), "total_scenarios": torch.tensor([len(num_h)]),
                                  "mean_hom_scenarios": torch.tensor([mean_hom_scenario]), "loss": torch.tensor([loss]),
                                  "baseline_accuracy": torch.tensor([baseline_accuracy])}
        return results
    
    def save_conflictive_scenarios(
        self,
        batch: AgentBatch,
        batch_id: int
    ) -> Union[List[Dict[str, Any]], Dict[AgentType, Dict[str, torch.Tensor]]]:
        """Predicts from a batch and then evaluates the output, returning the batched errors."""
        batch.to(self.device)

        # Run forward pass
        results: Dict[AgentType, Dict[str, torch.Tensor]] = dict()

        node_type: AgentType
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            mode = ModeKeys.PREDICT
            trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = self.request_homotopic_info(batch, get_real=True)
            enc = model.obtain_encoded_tensors(mode, agent_type_batch, h_tensor_combined)  
            pred = model.get_cost_from_encoders(enc, mode)
            success = np.array(success)
            for batch_i in range(len(batch.agent_fut)):
                if success[batch_i]:
                    goal_radius = 0.5
                    obstacles_vec = []
                    obstacles_previous = []
                    for obstacle_i in range(batch.neigh_fut.shape[1]):
                        if not batch.neigh_fut[batch_i, obstacle_i,0,0].isnan():
                            # pos_x = batch.neigh_fut[batch_i, obstacle_i, 0, 0] + torch.arange(-1, batch.agent_fut.shape[1], 
                            #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 2] * self.dt
                            # pos_y = batch.neigh_fut[batch_i, obstacle_i, 0, 1] + torch.arange(-1, batch.agent_fut.shape[1], 
                            #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 3] * self.dt
                            pos_x = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 0].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 0]])
                            pos_y = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 1].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 1]])
                            obstacles_vec.append(ObstacleMSG(id=obstacle_i, pos_x=pos_x, pos_y=pos_y, radius=self.person_radius))
                            obstacles_previous.append(ObstacleMSG(id=obstacle_i, pos_x=batch.neigh_hist[batch_i, obstacle_i, :, 0], pos_y=batch.neigh_hist[batch_i, obstacle_i, :, 1], radius=self.person_radius))
                    human_previous = ObstacleMSG(id=-1, pos_x=batch.agent_hist[batch_i, :, 0], pos_y=batch.agent_hist[batch_i, :, 1], radius=self.person_radius)
                    baseline_traj = ObstacleMSG(id=-1, pos_x=trajs_tensor_combined[batch_i, 0, :, 0], pos_y=trajs_tensor_combined[batch_i, 0, :, 1], radius=self.person_radius)
                    ours_traj = ObstacleMSG(id=-1, pos_x=trajs_tensor_combined[batch_i, torch.argmin(pred[batch_i, :num_h[batch_i]]), :, 0], 
                                            pos_y=trajs_tensor_combined[batch_i, torch.argmin(pred[batch_i, :num_h[batch_i]]), :, 1], radius=self.person_radius)
                    goal_x = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,0]
                    goals_x = [goal_x, goal_x + goal_radius, goal_x + goal_radius, goal_x - goal_radius, goal_x - goal_radius]
                    goal_y = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,1]
                    goals_y = [goal_y, goal_y + goal_radius, goal_y - goal_radius, goal_y + goal_radius, goal_y - goal_radius]  
                    if (torch.argmin(ground_truth[batch_i, :num_h[batch_i]]) == torch.argmin(pred[batch_i, :num_h[batch_i]]) and
                        torch.argmin(ground_truth[batch_i, :num_h[batch_i]]) == 0):
                            self.write_scenario_file_truth(batch_i, obstacles_vec, agent_type_batch, goals_x, goals_y, obstacles_previous,
                                                human_previous, baseline_traj, ours_traj, "both_right", batch_id)
                    elif (torch.argmin(ground_truth[batch_i, :num_h[batch_i]]) == torch.argmin(pred[batch_i, :num_h[batch_i]]) and
                        torch.argmin(ground_truth[batch_i, :num_h[batch_i]]) != 0):
                            self.write_scenario_file_truth(batch_i, obstacles_vec, agent_type_batch, goals_x, goals_y, obstacles_previous,
                                                human_previous, baseline_traj, ours_traj, "good_scenarios", batch_id)   
                    elif (torch.argmin(ground_truth[batch_i, :num_h[batch_i]]) == 0):
                            self.write_scenario_file_truth(batch_i, obstacles_vec, agent_type_batch, goals_x, goals_y, obstacles_previous,
                                                human_previous, baseline_traj, ours_traj, "bad_scenarios", batch_id)   
                    else:
                            self.write_scenario_file_truth(batch_i, obstacles_vec, agent_type_batch, goals_x, goals_y, obstacles_previous,
                                                human_previous, baseline_traj, ours_traj, "both_wrong", batch_id)           
            num_h = np.array(num_h)
            pred = pred[success]
            ground_truth = ground_truth[success]
            num_h = num_h[success]
            y_acc = []
            base_acc = []
            for i_acc in range(len(pred)):
                if num_h[i_acc] > 1:
                    y_acc.append(torch.argmin(pred[i_acc, :num_h[i_acc]]) == torch.argmin(ground_truth[i_acc, :num_h[i_acc]]))
                    base_acc.append(torch.argmin(ground_truth[i_acc, :num_h[i_acc]])==0)
            acc = sum(y_acc)/len(y_acc)
            baseline_accuracy = sum(base_acc)/len(y_acc)
            mean_hom_scenario = sum(num_h[num_h>1])/sum(num_h>1)
            loss = self.loss_function(pred[ground_truth!=-1], ground_truth[ground_truth!=-1])
            # batch_eval: Dict[
            #     str, torch.Tensor
            # ] = evaluation.compute_batch_statistics_pt(
            #     agent_type_batch.agent_fut[..., :2],
            #     prediction_output_dict=predictions,
            #     y_dists=y_dists,
            # )
            results[node_type] = {"accuracy": torch.tensor([acc]), "scenarios_homologies": torch.tensor([sum(num_h>1)]), 
                                  "failed_scenarios": torch.tensor([sum(np.logical_not(success))]), "total_scenarios": torch.tensor([len(num_h)]),
                                  "mean_hom_scenarios": torch.tensor([mean_hom_scenario]), "loss": torch.tensor([loss]),
                                  "baseline_accuracy": torch.tensor([baseline_accuracy])}
        return results
    
    def visualize_scenarios(
        self,
        batch: AgentBatch,
        batch_id: int
    ) -> Union[List[Dict[str, Any]], Dict[AgentType, Dict[str, torch.Tensor]]]:
        """Predicts from a batch and then evaluates the output, returning the batched errors."""
        batch.to(self.device)

        # Run forward pass
        results: Dict[AgentType, Dict[str, torch.Tensor]] = dict()

        node_type: AgentType
        for node_type in batch.agent_types():
            model: EncoderWithH = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)
            mode = ModeKeys.PREDICT
            for batch_i in range(len(batch.agent_fut)):
                batch_aux = copy.deepcopy(agent_type_batch)
                batch_aux.agent_fut = batch_aux.agent_fut[batch_i,:,:].unsqueeze(0)
                batch_aux.agent_fut_extent = batch_aux.agent_fut_extent[batch_i,:,:].unsqueeze(0)
                batch_aux.agent_hist = batch_aux.agent_hist[batch_i,:,:].unsqueeze(0)
                batch_aux.agent_hist_extent = batch_aux.agent_hist_extent[batch_i,:,:].unsqueeze(0)
                batch_aux.neigh_fut = batch_aux.neigh_fut[batch_i,:,:,:].unsqueeze(0)
                batch_aux.neigh_fut_extents = batch_aux.neigh_fut_extents[batch_i,:,:,:].unsqueeze(0)
                batch_aux.neigh_hist = batch_aux.neigh_hist[batch_i,:,:,:].unsqueeze(0)
                batch_aux.neigh_hist_extents = batch_aux.neigh_hist_extents[batch_i,:,:,:].unsqueeze(0)
                batch_aux.agent_fut_len = batch_aux.agent_fut_len[batch_i].unsqueeze(0)
                batch_aux.agent_hist_len = batch_aux.agent_hist_len[batch_i].unsqueeze(0)
                batch_aux.neigh_fut_len = batch_aux.neigh_fut_len[batch_i].unsqueeze(0)
                batch_aux.neigh_hist_len = batch_aux.neigh_hist_len[batch_i].unsqueeze(0)
                batch_aux.num_neigh = batch_aux.num_neigh[batch_i].unsqueeze(0)
                batch_aux.neigh_types = batch_aux.neigh_types[batch_i].unsqueeze(0)
                trajs_tensor_combined, ground_truth, success, num_h, h_tensor_combined = self.request_homotopic_info(batch_aux, get_real=True)
                enc = model.obtain_encoded_tensors(mode, batch_aux, h_tensor_combined)  
                pred = model.get_cost_from_encoders(enc, mode)
                success = np.array(success)
                if success[0]:
                    goal_radius = 0.5
                    obstacles_vec = []
                    obstacles_previous = []
                    for obstacle_i in range(batch.neigh_fut.shape[1]):
                        if not batch.neigh_fut[batch_i, obstacle_i,0,0].isnan():
                            # pos_x = batch.neigh_fut[batch_i, obstacle_i, 0, 0] + torch.arange(-1, batch.agent_fut.shape[1], 
                            #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 2] * self.dt
                            # pos_y = batch.neigh_fut[batch_i, obstacle_i, 0, 1] + torch.arange(-1, batch.agent_fut.shape[1], 
                            #             device=batch.neigh_hist.device) * batch.neigh_fut[batch_i, obstacle_i, 0, 3] * self.dt
                            pos_x = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 0].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 0]])
                            pos_y = torch.concat([batch.neigh_hist[batch_i, obstacle_i, -1, 1].unsqueeze(0), batch.neigh_fut[batch_i, obstacle_i, :, 1]])
                            obstacles_vec.append(ObstacleMSG(id=obstacle_i, pos_x=pos_x, pos_y=pos_y, radius=self.person_radius))
                            obstacles_previous.append(ObstacleMSG(id=obstacle_i, pos_x=batch.neigh_hist[batch_i, obstacle_i, :, 0], pos_y=batch.neigh_hist[batch_i, obstacle_i, :, 1], radius=self.person_radius))
                    human_previous = ObstacleMSG(id=-1, pos_x=batch.agent_hist[batch_i, :, 0], pos_y=batch.agent_hist[batch_i, :, 1], radius=self.person_radius)
                    baseline_traj = ObstacleMSG(id=-1, pos_x=trajs_tensor_combined[0, 0, :, 0], pos_y=trajs_tensor_combined[0, 0, :, 1], radius=self.person_radius)
                    ours_traj = ObstacleMSG(id=-1, pos_x=trajs_tensor_combined[0, torch.argmin(pred[0, :num_h[0]]), :, 0], 
                                            pos_y=trajs_tensor_combined[0, torch.argmin(pred[0, :num_h[0]]), :, 1], radius=self.person_radius)
                    goal_x = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,0]
                    goals_x = [goal_x, goal_x + goal_radius, goal_x + goal_radius, goal_x - goal_radius, goal_x - goal_radius]
                    goal_y = batch.agent_fut[batch_i,batch.agent_fut_len[batch_i]-1,1]
                    goals_y = [goal_y, goal_y + goal_radius, goal_y - goal_radius, goal_y + goal_radius, goal_y - goal_radius]  
                    human_x_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,0]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],0]])
                    human_y_traj = torch.concat([torch.tensor([batch.agent_hist[batch_i,-1,1]], device=self.device), batch.agent_fut[batch_i,:batch.agent_fut_len[batch_i],1]])                   
                    if torch.argmin(ground_truth[0, :num_h[0]]) != 0 or torch.argmin(pred[0, :num_h[0]]) != 0: 
                        print("BATCH_i:", batch_i)
                        print("Ground truth: ", torch.argmin(ground_truth[0, :num_h[0]]), "Ours: ", torch.argmin(pred[0, :num_h[0]]))                            
                        print(ground_truth, pred, num_h)
                        plot_scenario_dynamic(obstacles_previous_in=obstacles_previous, obstacles_vec_in=obstacles_vec, goals_in=ObstacleMSG(id=-1, pos_x=goals_x, pos_y=goals_y, radius=0.1),
                                            human_truth_in=ObstacleMSG(id=-1, pos_x=human_x_traj, pos_y=human_y_traj, radius=self.person_radius), human_previous_in=human_previous,
                                            baseline_traj_in=baseline_traj, ours_traj_in=ours_traj)
        return {}
    
    def get_topologies(self, obstacles, x, y, orientation, v, goals_x, goals_y, static_x, static_y, static_n, n_trajectories):
        req = guidancesRequest()
        req.obstacles = obstacles
        req.x = 0.0
        req.y = 0.0
        req.oriantation = 0.0
        req.v = 0.0
        req.goals_x = goals_x
        req.goals_y = goals_y
        req.static_x = static_x
        req.static_y = static_y
        req.static_n = static_n
        req.n_trajectories = n_trajectories
        guidances_srv = rospy.ServiceProxy('/get_guidances', guidances)
        res = guidances_srv(req)
        return res.success, res.trajectories, res.h_signature
    
    def get_topologies_Hcost(self, obstacles, x, y, orientation, v, goals_x, goals_y, static_x, static_y, static_n, n_trajectories,
                                    truth, data_idx, check_stored):
        if check_stored and data_idx[0] in self.topologies_dict:
            res = self.topologies_dict[data_idx[0]]
        else:
            req = guidances_HcostRequest()
            req.obstacles = obstacles
            req.x = 0.0
            req.y = 0.0
            req.oriantation = 0.0
            req.v = 0.0
            req.goals_x = goals_x
            req.goals_y = goals_y
            req.static_x = static_x
            req.static_y = static_y
            req.static_n = static_n
            req.n_trajectories = n_trajectories
            truth.x[0] = 0.0
            truth.y[0] = 0.0
            req.truth = truth
            guidances_srv = rospy.ServiceProxy('/get_guidances_Hcost', guidances_Hcost)
            res = guidances_srv(req)
            # self.topologies_dict[data_idx[0]] = res
        return res.success, res.trajectories, res.costs, res.h_signature
