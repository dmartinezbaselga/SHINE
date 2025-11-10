#!/bin/bash
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

### NOTE: These commands are meant to serve as templates, meaning some of the values are truncated for readability.
### To exactly reproduce model training from our paper, please see kf_models/**/config.json for full-fidelity values.
### To use these commands on datasets other than eupeds_eth, make sure to change the `train_data`, `eval_data`,
### and `data_loc_dict` arguments.

### Q: Why 2.8s of history length instead of 3.2 s as used in other works?
### A: While it is oft-said in many papers (e.g., S-GAN, Trajectron++, and nearly all following works)
###    that 3.2 s = 8 timesteps of observation (or history) are used as input,
###    in reality these approaches use 8 _observations_ (which naively comes out to 3.2s via 8*0.4s).
###    However, this neglects the fact that there are only 7 time _steps_ between those 8 points,
###    yielding only 2.8 s of elapsed time. Thus, in this script, you will see us using 2.8s of
###    history to predict 4.8s of future motion, which comes out to using 8 observed timesteps
###    (7 history + 1 current) to predict 12 future timesteps.
###    To summarize, it is because 8 observations = 7 historical timesteps + the current timestep.

# Base
# torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_base --train_epochs=5 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train --eval_data=eupeds_eth-val --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.016 --sigma_eps_init=0.0002
# torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_cost --train_epochs=5 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train --eval_data=eupeds_eth-val --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.016 --sigma_eps_init=0.0002
#ETH
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_hcost_eth --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_eth\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_eth-train_loo --eval_data=eupeds_eth-val_loo --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.0001 --sigma_eps_init=0.0002
#Hotel
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_hcost_hotel --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_hotel\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_hotel-train_loo --eval_data=eupeds_hotel-val_loo --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.0001 --sigma_eps_init=0.0002
#Univ
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_hcost_univ --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_univ\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_univ-train_loo --eval_data=eupeds_univ-val_loo --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.0001 --sigma_eps_init=0.0002
#Zara1
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_hcost_zara1 --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_zara1\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_zara1-train_loo --eval_data=eupeds_zara1-val_loo --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.0001 --sigma_eps_init=0.0002
#Zara2
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=eth_guidance_hcost_zara2 --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_zara2\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=eupeds_zara2-train_loo --eval_data=eupeds_zara2-val_loo --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.0001 --sigma_eps_init=0.0002

# Full model
torchrun --nproc_per_node=1 --master_port=29500 ../../train_guidance.py --save_every=1 --eval_every=1 --vis_every=1 --batch_size=32 --eval_batch_size=32 --preprocess_workers=16 --log_dir=kf_models --log_tag=full_model_guidance_hcost --train_epochs=10 --conf=../../config/pedestrians.json --trajdata_cache_dir=~/.unified_data_cache --data_loc_dict=\{\"eupeds_zara2\":\ \"~/datasets/eth_ucy_peds\"\} --train_data=full_model --eval_data=full_model --history_sec=2.8 --prediction_sec=4.8 --K=1 --alpha_init=1.54e-05 --augment_input_noise=0.57 --grad_clip=0.909 --learning_rate=0.001 --sigma_eps_init=0.0002

# Oracle is just Base on the target split
