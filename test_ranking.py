import os
import torch
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
matplotlib.rcParams.update({'font.size': 18})
supported_labels = [
    "random",
    "local_search",
    "ei",
    "ei_ego_network_1",
    "ei_ego_network_1_old",
    "dfs",
    "bfs",
    "ei_ego_network_2",
    "ei_ego_network_2_no_ard",
]
# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))
d_color = {"ei_ego_network_1":"#1f77b4", "ei_ego_network_2":"#8c564b", "random":"#ff7f0e",
           "local_search":"#2ca02c", "dfs": "#d62728", "bfs": "#9467bd", "ei_ego_network_1_old":"#e377c2",
           "ei_ego_network_2_no_ard":"#7f7f7f"}
d_label = {"ei_ego_network_1":"BO_Poly", "ei_ego_network_1_old":"BO_SumInverse",
           "ei_ego_network_2":"BO_Diff_ARD", "ei_ego_network_2_no_ard":"BO_Diff", 
           "random":"Random", "local_search":"Local search", "dfs": "Dfs", "bfs": "Bfs"}

algorithm_names = ["bfs", "dfs", "ei_ego_network_1", "ei_ego_network_1_old", "ei_ego_network_2", "ei_ego_network_2_no_ard", "local_search", "random"]
N_SEED, N_ITER = 10, 100
def plot_result(list_experiences):
    d_algorithm = {name:np.empty((0,N_SEED, N_ITER)) for name in algorithm_names}
    for el in list_experiences:
        path, list_label = el[0], el[1]
        # list_label contains the list of experiments in path
        for label in list_label:
            experiment_dict = {name:np.array([]) for name in algorithm_names}
            # path: main directory of experiments (eg. logs/centrality), label : label experiment to plot
            exp_dir = os.path.join(path, label)
            for algorithm in algorithm_names:
                alg_dir = os.path.join(exp_dir, algorithm)
                ## Here are in directory with signal png and pt
                data_path_seeds = sorted([f for f in os.listdir(alg_dir) if ".pt" in f])
                data_over_seeds = []
                for i, df in enumerate(data_path_seeds):
                    data_path = os.path.join(alg_dir, df)
                    data = torch.load(data_path, map_location="cpu")
                    assert "Y" in data.keys()
                    y = data["Y"].numpy().flatten()
                    data_over_seeds.append(y)
                #max_len = max(n_data_per_trial)
                max_len = N_ITER
                for i, d in enumerate(data_over_seeds):
                    if d.shape[0] < max_len:
                        data_over_seeds[i] = np.concatenate((d, d[-1] * np.ones(max_len - d.shape[0])))
                    elif d.shape[0] > max_len:
                        data_over_seeds[i] = d[:max_len]
                all_data = np.array(data_over_seeds)
                y = pd.DataFrame(all_data).cummax(axis=1).to_numpy()
                experiment_dict[algorithm] = y
            
            ### Concatenate here
            for i, alg_name in enumerate(algorithm_names):
                d_algorithm[alg_name] = np.vstack((d_algorithm[alg_name], [experiment_dict[alg_name]])) # contains, for each algo, (n_exp) x n_seed x n_iter tensors
    
    ## Take mean std
    s = d_algorithm[algorithm_names[0]].shape
    tensor_to_rank = np.empty((0, s[0]*s[1], s[2]))
    for alg_name in algorithm_names:
        values = d_algorithm[alg_name]
        values = values.reshape(-1, values.shape[-1]) #averaging accross seeds and experiments 
        tensor_to_rank = np.vstack((tensor_to_rank, [values]))
    rank_tensor = (-tensor_to_rank).argsort(axis=0).argsort(axis=0)
    mean = rank_tensor.mean(axis=1)
    lb = mean - rank_tensor.std(axis=1)/np.sqrt(rank_tensor.shape[1])
    ub = mean + rank_tensor.std(axis=1)/np.sqrt(rank_tensor.shape[1])
    for i, alg_name in enumerate(algorithm_names):
        d_algorithm[alg_name] = [mean[i], lb[i], ub[i]]
    return d_algorithm

path = "./logs/centrality/"
list_experiences = ["ba_betweenness_m-2_ninit-20_n-1000", "ba_betweenness_m-3_ninit-20_n-1000", "ba_betweenness_m-4_ninit-20_n-1000",
                    "ba_eigenvector_centrality_m-2_ninit-20_n-1000", "ba_eigenvector_centrality_m-3_ninit-20_n-1000", "ba_eigenvector_centrality_m-4_ninit-20_n-1000"]

d_algorithm = plot_result([[path, list_experiences]])
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(N_ITER)
for alg in algorithm_names:
    result_algo = d_algorithm[alg]
    ax.plot(x, result_algo[0], ".-", label=d_label[alg], color=d_color[alg])
    ax.fill_between(x, result_algo[1], result_algo[2], alpha=0.1, color=d_color[alg])
ax.legend()
plt.show()