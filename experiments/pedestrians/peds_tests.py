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

from torch import optim, nn
from torch.utils import data
from tqdm.notebook import tqdm
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron
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

    model_registrar = ModelRegistrar(model_dir, device)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
        
    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams

# %%
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.set_device(0)
else:
    device = 'cpu'
print(torch.cuda.is_available())

# %%
def finetune_update(model: Trajectron, batch: AgentBatch = None, dataloader: data.DataLoader = None, num_epochs: int = None, update_mode: UpdateMode = UpdateMode.NO_UPDATE) -> float:
    if batch is None and dataloader is None:
        raise ValueError("Only one of batch or dataloader can be passed in.")
    
    if dataloader is not None and num_epochs is None:
        raise ValueError("num_epochs must not be None if dataloader is not None.")
    
    lr_scheduler = None
    optimizer = optim.Adam([{'params': model.model_registrar.get_all_but_name_match('map_encoder').parameters()},
                            {'params': model.model_registrar.get_name_match('map_encoder').parameters(),
                             'lr': model.hyperparams['map_enc_learning_rate']/10}],
                           lr=model.hyperparams['learning_rate']/10)
    # Set Learning Rate
    if model.hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif model.hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=model.hyperparams['learning_decay_rate'])
    
    if batch is not None:
        model.step_annealers()
        optimizer.zero_grad(set_to_none=True)

        train_loss = model(batch, update_mode=update_mode)
        train_loss.backward()

        # Clipping gradients.
        if model.hyperparams['grad_clip'] is not None:
            nn.utils.clip_grad_value_(model.model_registrar.parameters(), model.hyperparams['grad_clip'])
        
        optimizer.step()
        
        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
    
    elif dataloader is not None:
        batch: AgentBatch
        for batch_idx, batch in enumerate(dataloader):
            model.step_annealers()
            
            optimizer.zero_grad(set_to_none=True)

            train_loss = model(batch)
                        
            train_loss.backward()

            # Clipping gradients.
            if model.hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model.model_registrar.parameters(), model.hyperparams['grad_clip'])
            
            optimizer.step()
            
            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()

# %%
def finetune_last_layer_update(model: Trajectron, batch: AgentBatch = None, dataloader: data.DataLoader = None, num_epochs: int = None) -> float:
    if batch is None and dataloader is None:
        raise ValueError("Only one of batch or dataloader can be passed in.")
    
    if dataloader is not None and num_epochs is None:
        raise ValueError("num_epochs must not be None if dataloader is not None.")
    
    lr_scheduler = None
    optimizer = optim.Adam([{'params': model.model_registrar.get_all_but_name_match('last_layer').parameters()},
                            {'params': model.model_registrar.get_name_match('last_layer').parameters(),
                             'lr': model.hyperparams['learning_rate']/10}],
                           lr=0)
    # Set Learning Rate
    if model.hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif model.hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=model.hyperparams['learning_decay_rate'])
    
    if batch is not None:
        model.step_annealers()
        optimizer.zero_grad(set_to_none=True)

        train_loss = model(batch)
        train_loss.backward()

        # Clipping gradients.
        if model.hyperparams['grad_clip'] is not None:
            nn.utils.clip_grad_value_(model.model_registrar.parameters(), model.hyperparams['grad_clip'])
        
        optimizer.step()
        
        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
            
    elif dataloader is not None:
        batch: AgentBatch
        for batch_idx, batch in enumerate(dataloader):
            model.step_annealers()
            
            optimizer.zero_grad(set_to_none=True)

            train_loss = model(batch)
                        
            train_loss.backward()

            # Clipping gradients.
            if model.hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model.model_registrar.parameters(), model.hyperparams['grad_clip'])
            
            optimizer.step()
            
            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()

# %%
peds_datasets = ["eth", "hotel", "univ", "zara1", "zara2"]

history_sec = 2.8
prediction_sec = 4.8

base_checkpoint = 5
k0_checkpoint = 4
adaptive_checkpoint = 5
oracle_checkpoint = 5

train_scene = "zara1"
eval_scene = "zara1"

base_model = glob.glob(f"kf_models/{train_scene}_*_base_*")[0]

eval_dataset = f"eupeds_{eval_scene}"
eval_data = f"{eval_dataset}-test_loo"

base_trajectron, _ = load_model(base_model, device, epoch=base_checkpoint,
    custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                        "single_mode_multi_sample": False}
)

# Load training and evaluation environments and scenes
attention_radius = defaultdict(lambda: 20.0) # Default range is 20m unless otherwise specified.
attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

online_eval_dataset = UnifiedDataset(
    desired_data=[eval_data],
    history_sec=(0.1, history_sec),
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

# %%
prog = re.compile("(.*)/(?P<scene_name>.*)/(.*)$")

def plot_outputs(
    eval_dataset: UnifiedDataset,
    dataset_idx: int,
    model: Trajectron,
    model_name: str,
    agent_ts: int,
    save=True,
    extra_str=None,
    subfolder="",
    filetype="png"
):
    batch: AgentBatch = eval_dataset.get_collate_fn(pad_format="right")([eval_dataset[dataset_idx]])
    
    fig, ax = plt.subplots()
    trajdata_vis.plot_agent_batch(batch, batch_idx=0, ax=ax, show=False, close=False)
    
    with torch.no_grad():
        # predictions = model.predict(batch,
        #                             z_mode=True,
        #                             gmm_mode=True,
        #                             full_dist=False,
        #                             output_dists=False)
        # prediction = next(iter(predictions.values()))
        
        pred_dists, _ = model.predict(batch,
                                      z_mode=False,
                                      gmm_mode=False,
                                      full_dist=True,
                                      output_dists=True)
        # pred_dist = next(iter(pred_dists.values()))

    visualization.visualize_distribution(ax, pred_dists, batch_idx=0)
    
    # batch_eval: Dict[str, torch.Tensor] = evaluation.compute_batch_statistics_pt(
    #     batch.agent_fut[..., :2],
    #     prediction_output_dict=torch.from_numpy(prediction),
    #     y_dists=pred_dist
    # )
    
    scene_info_path, _, scene_ts = eval_dataset._data_index[dataset_idx]
    scene_name = prog.match(scene_info_path).group("scene_name")
    
    agent_name = batch.agent_name[0]
    agent_type_name = f"{str(AgentType(batch.agent_type[0].item()))}/{agent_name}"
    
    ax.set_title(f"{scene_name}/t={scene_ts} {agent_type_name}")
    # print(model_name, extra_str, batch_eval)
    
    if save:
        fname = f"plots/{subfolder}{model_name}_{scene_name}_{agent_name}_t{agent_ts}"
        if extra_str:
            fname += "_" + extra_str
        fig.savefig(fname + f".{filetype}")
        
        plt.close(fig)

# %%
def get_dataloader(
    eval_dataset: UnifiedDataset,
    batch_size: int = 128,
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
metrics_list = ['ml_ade', 'ml_fde', 'min_ade_5', 'min_fde_5', 'min_ade_10', 'min_fde_10', 'nll_mean', 'nll_final']

# %% [markdown]
# ### Qualitative Plots

# %%
plot_outputs(batch_eval_dataset, 200, base_trajectron, "Base", -1, subfolder=f"{train_scene}2{eval_scene}_", filetype="pdf")

# %% [markdown]
# ### Offline Evaluation

# %%
def add_results_to_summary(
    model_name: str,
    overall_summary_dict: Dict[str, List[float]],
    model_perf_dict: Dict[AgentType, Dict[str, np.ndarray]]
):
    overall_summary_dict["model"].append(model_name)
    for metric in metrics_list:
        overall_summary_dict[metric].append(np.concatenate(model_perf_dict[AgentType.PEDESTRIAN][metric]).mean().item())

# %%
eval_dict = defaultdict(list)

with torch.no_grad():
    # Base Model
    batch_eval_dataloader = get_dataloader(batch_eval_dataset, num_workers=16)
    model_perf = defaultdict(lambda: defaultdict(list))
    for eval_batch in tqdm(batch_eval_dataloader, desc=f'Eval Base'):
        eval_results: Dict[AgentType, Dict[str, torch.Tensor]] = base_trajectron.predict_and_evaluate_batch(eval_batch)
        for agent_type, metric_dict in eval_results.items():
            for metric, values in metric_dict.items():
                model_perf[agent_type][metric].append(values.cpu().numpy())

    add_results_to_summary("Base", eval_dict, model_perf)
    print(model_perf)
    del eval_batch
    #del eval_results
    #del model_perf

# %%
eval_dict

# %%


# %%
print(eval_results[2].keys())

# %%
with open(f"results/{train_scene}2{eval_scene}_model_performance.pkl", 'wb') as f:
    pickle.dump(eval_dict, f)

# %%
with open(f"results/{train_scene}2{eval_scene}_model_performance.pkl", 'rb') as f:
    eval_dict = pickle.load(f)

# %%
eval_dict

# %% [markdown]
# ### Dataset2Dataset

# %%
def add_results_to_summary(
    model_name: str,
    scene_from: str,
    scene_to: str,
    overall_summary_dict: Dict[str, List[float]],
    model_perf_dict: Dict[AgentType, Dict[str, np.ndarray]]
):
    overall_summary_dict["model"].append(model_name)
    overall_summary_dict["scene_from"].append(scene_from)
    overall_summary_dict["scene_to"].append(scene_to)
    for metric in ["ml_ade", "ml_fde"]:
        overall_summary_dict[metric].append(np.concatenate(model_perf_dict[AgentType.PEDESTRIAN][metric]).mean().item())

# %%
peds_datasets = ["eth", "hotel", "univ", "zara1", "zara2"]

eval_dict = defaultdict(list)

eval_dict["model"] = ["T-GNN"] * 20
for train_scene in peds_datasets:
    for eval_scene in [ds for ds in peds_datasets if ds != train_scene]:
        eval_dict["scene_from"].append(train_scene)
        eval_dict["scene_to"].append(eval_scene)

# From T-GNN paper: https://arxiv.org/abs/2203.05046
eval_dict["ml_ade"].extend([1.13, 1.25, 0.94, 1.03, 2.54, 1.08, 2.25, 1.41, 0.97, 0.54, 0.61, 0.23, 0.88, 0.78, 0.59, 0.32, 0.87, 0.72, 0.65, 0.34])
eval_dict["ml_fde"].extend([2.18, 2.25, 1.78, 1.84, 4.15, 1.82, 4.04, 2.53, 1.91, 1.12, 1.30, 0.87, 1.92, 1.46, 1.25, 0.65, 1.86, 1.45, 1.28, 0.72])

# %% [markdown]
# ## Online (Per Agent)

# %%
def per_agent_eval(
    curr_agent: str,
    model: Trajectron,
    model_name: str,
    batch: AgentBatch,
    agent_ts: int,
    model_eval_dict: DefaultDict[str, Union[List[int], List[float]]],
    plot=True
):
    with torch.no_grad():
        if plot:
            plot_outputs(online_eval_dataset,
                         dataset_idx=batch.data_idx[0].item(),
                         model=model,
                         model_name=model_name,
                         agent_ts=agent_ts,
                         subfolder=f"per_agent_{train_scene}2{eval_scene}/")
        
        model_perf = defaultdict(lambda: defaultdict(list))
        eval_results: Dict[AgentType, Dict[str, torch.Tensor]] = model.predict_and_evaluate_batch(batch)
        for agent_type, metric_dict in eval_results.items():
            for metric, values in metric_dict.items():
                model_perf[agent_type][metric].append(values.cpu().numpy())

        for idx, metric in enumerate(metrics_list):
            if len(model_perf[AgentType.PEDESTRIAN]) == 0:
                break
            
            metric_values = np.concatenate(model_perf[AgentType.PEDESTRIAN][metric]).tolist()
            if idx == 0:
                model_eval_dict["agent_ts"].extend([agent_ts] * len(metric_values))

            model_eval_dict[metric].extend(metric_values)

# %%
def init_time_eval(
    curr_agent: str,
    model: Trajectron,
    model_name: str,
    online_batch: AgentBatch,
    model_eval_dict: DefaultDict[str, Union[List[int], List[float]]],
    plot=True
):
    if plot:
        plot_outputs(online_eval_dataset,
                     dataset_idx=online_batch.data_idx[0].item(),
                     model=model,
                     model_name=model_name,
                     agent_ts=0,
                     subfolder=f"per_agent_{train_scene}2{eval_scene}/",
                     extra_str=f"init")
    per_agent_eval(curr_agent, model, model_name, online_batch, 0, model_eval_dict, plot=False)

# %%
def per_agent_s2s(
    online_eval_dataset, adaptive_trajectron,
    base_trajectron, k0_trajectron, oracle_trajectron,
    base_model, base_checkpoint, k0_model, k0_checkpoint,
    n_samples=3001
):
    adaptive_dict = defaultdict(list)
    finetune_dict = defaultdict(list)
    k0_finetune_dict = defaultdict(list)

    base_dict = defaultdict(list)
    k0_dict = defaultdict(list)
    oracle_dict = defaultdict(list)

    online_eval_dataloader = get_dataloader(online_eval_dataset, batch_size=1, num_workers=4, shuffle=False)

    adaptive_trajectron.reset_adaptive_info()

    outer_pbar = tqdm(
        online_eval_dataloader,
        total=min(n_samples, len(online_eval_dataloader)),
        desc=f'Adaptive Eval PH={prediction_sec}',
        position=0,
    )

    plot_per_step = False

    curr_agent: str = None
    agent_ts: int = 0
    online_batch: AgentBatch
    for data_sample, online_batch in enumerate(outer_pbar):
        if data_sample >= n_samples:
            outer_pbar.close()
            break

        if online_batch.agent_name[0] != curr_agent:
            # Resetting the K_n, L_n for each Bayesian last layer.
            adaptive_trajectron.reset_adaptive_info()

            # Resetting the finetune baseline to its base.
            finetune_trajectron, _ = load_model(base_model, device, epoch=base_checkpoint,
                custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                                    "single_mode_multi_sample": False}
            )
            k0_finetune_trajectron, _ = load_model(k0_model, device, epoch=k0_checkpoint,
                custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                                    "single_mode_multi_sample": False}
            )

            curr_agent = online_batch.agent_name[0]
            agent_ts: int = 0

#             init_time_eval(curr_agent, adaptive_trajectron, "Ours", online_batch, adaptive_dict, plot_per_step)
#             init_time_eval(curr_agent, finetune_trajectron, "Finetune", online_batch, finetune_dict, plot_per_step)
#             init_time_eval(curr_agent, k0_finetune_trajectron, "K0+Finetune", online_batch, k0_finetune_dict, plot_per_step)

#             init_time_eval(curr_agent, base_trajectron, "Base", online_batch, base_dict, plot_per_step)
#             init_time_eval(curr_agent, k0_trajectron, "K0", online_batch, k0_dict, plot_per_step)
#             init_time_eval(curr_agent, oracle_trajectron, "Oracle", online_batch, oracle_dict, plot_per_step)

        with torch.no_grad():
            # This is the inference call that internally updates L_n and K_n.
            adaptive_trajectron.adaptive_predict(
                online_batch,
                update_mode=UpdateMode.ITERATIVE
            )

        finetune_update(finetune_trajectron, online_batch)
        finetune_last_layer_update(k0_finetune_trajectron, online_batch)

        # # This is effectively measuring number of updates/observed data points.
        # agent_ts += 1

        if agent_ts % 10 == 0:
            per_agent_eval(curr_agent, adaptive_trajectron, "Ours", online_batch, agent_ts, adaptive_dict, plot=plot_per_step)
            per_agent_eval(curr_agent, finetune_trajectron, "Finetune", online_batch, agent_ts, finetune_dict, plot=plot_per_step)
            per_agent_eval(curr_agent, k0_finetune_trajectron, "K0+Finetune", online_batch, agent_ts, k0_finetune_dict, plot=plot_per_step)

            per_agent_eval(curr_agent, base_trajectron, "Base", online_batch, agent_ts, base_dict, plot=plot_per_step)
            per_agent_eval(curr_agent, k0_trajectron, "K0", online_batch, agent_ts, k0_dict, plot=plot_per_step)
            per_agent_eval(curr_agent, oracle_trajectron, "Oracle", online_batch, agent_ts, oracle_dict, plot=plot_per_step)

        # This is effectively measuring the most-recently seen timestep.
        agent_ts += 1
        
    return (
        adaptive_dict, finetune_dict, k0_finetune_dict,
        base_dict, k0_dict, oracle_dict
    )

# %%
def relativize_df(input_df, reference_df):
    output_df = input_df.copy()
    output_df.loc[:, metrics_list] = 100*(output_df.loc[:, metrics_list] - reference_df.loc[:, metrics_list])/reference_df.loc[:, metrics_list]
    return output_df

# %%
import matplotlib.ticker as mtick

combined_df = pd.concat((
    relativize_df(adaptive_eval_df, base_eval_df),
    relativize_df(finetune_eval_df, base_eval_df),
    relativize_df(k0_finetune_eval_df, base_eval_df),
    
    relativize_df(k0_eval_df, base_eval_df),
    relativize_df(base_eval_df, base_eval_df),
    relativize_df(oracle_eval_df, base_eval_df)
), ignore_index=True)

combined_df.to_csv(f"results/kf_{train_scene}2{eval_scene}_per_agent_online_rel.csv", index=False)

# %%



