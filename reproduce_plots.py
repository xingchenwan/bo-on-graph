import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
matplotlib.rcParams.update({'font.size': 15})
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
           "ei_ego_network_2_no_ard":"#7f7f7f", "ei_ego_network_matern":'#bcbd22'}
d_label = {"ei_ego_network_1":"BO_Poly", "ei_ego_network_1_old":"BO_SumInverse",
           "ei_ego_network_2":"BO_Diff_ARD", "ei_ego_network_2_no_ard":"BO_Diff", 
           "random":"Random", "local_search":"Local search", "dfs": "Dfs", "bfs": "Bfs",
           "ei_ego_network_matern":"BO_Matern"}

def plot_result(path: str, label: str, plot_kwargs: dict = None, median=False, cumulative=False, regret=True):
    plot_kwargs = deepcopy(plot_kwargs) or {}
    data_path_seeds = [f for f in os.listdir(path) if ".pt" in f]
    data_over_seeds = []
    for i, df in enumerate(data_path_seeds):
        data_path = os.path.join(path, df)
        with open(data_path, "rb") as fp:
            data = torch.load(data_path, map_location="cpu")
            minimize = False
        if "regret" in data.keys() and regret:
            y = -data["regret"].numpy().flatten()   # to maximize negative regret
            minimize = True
        else:
            assert "Y" in data.keys()
            y = data["Y"].numpy().flatten()
        data_over_seeds.append(y)
    n_data_per_trial = np.array([len(d) for d in data_over_seeds])
    print(path)
    max_len = max(n_data_per_trial)
    if len(np.unique(n_data_per_trial)) > 1:
        # pad as appropriate
        for i, d in enumerate(data_over_seeds):
            data_over_seeds[i] = np.concatenate((
                d, d[-1] * np.ones(max_len - d.shape[0])))
    all_data = np.array(data_over_seeds)
    if cumulative:
        y = pd.DataFrame(all_data).cummax(axis=1)
    else:
        y = pd.DataFrame(all_data)
    x = np.arange(all_data.shape[1])
    if median:
        mean = y.median(axis=0)
        lb = y.quantile(q=0.25, axis=0)
        ub = y.quantile(q=0.75, axis=0)
    else:
        mean = y.mean(axis=0)
        # standard error
        lb = mean - y.std(axis=0) / np.sqrt(all_data.shape[0])
        ub = mean + y.std(axis=0) / np.sqrt(all_data.shape[0])
    if minimize:
        mean = -mean
        lb = -lb
        ub = -ub
    plt.plot(x, mean, ".-", label=d_label[label], color=d_color[label], **plot_kwargs)
    if "alpha" in plot_kwargs.keys():
        del plot_kwargs["alpha"]
    if "markevery" in plot_kwargs.keys():
        del plot_kwargs["markevery"]
    plt.fill_between(x, lb, ub, alpha=0.1, color=d_color[label], **plot_kwargs)
    return y, max_len

if __name__ == "__main__":
    logs_dir = './logs'
    task_list = [name for name in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, name))]
    for task in task_list:
        task_dir = os.path.join(logs_dir, task)
        exp_name = [name for name in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, name))]
        for experiment in exp_name:
            exp_dir = os.path.join(task_dir, experiment)
            algorithm_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]
            min_max_len = np.inf
            try:
                for algorithm in algorithm_name:
                    alg_dir = os.path.join(exp_dir, algorithm)
                    ## Here are in directory with signal png and pt
                    y, max_len = plot_result(alg_dir, label=algorithm, median=False, cumulative=True)
                    min_max_len = min(min_max_len, max_len)
                print(min_max_len)
                plt.legend()
                plt.xlabel("#Iters")
                plt.ylabel("Objective")
                plt.xlim([0, min_max_len])
                #plt.yscale("log")
                plt.savefig(os.path.join(exp_dir, "plot_result_regretpng.png"), bbox_inches='tight')
                plt.savefig(os.path.join(exp_dir, "plot_result_regretpdf.pdf"), bbox_inches='tight')
                plt.clf()
            except:
                continue

    logs_dir = './logs'
    task_list = [name for name in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, name))]
    for task in task_list:
        task_dir = os.path.join(logs_dir, task)
        exp_name = [name for name in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, name))]
        for experiment in exp_name:
            exp_dir = os.path.join(task_dir, experiment)
            algorithm_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]
            min_max_len = np.inf
            try:
                for algorithm in algorithm_name:
                    alg_dir = os.path.join(exp_dir, algorithm)
                    ## Here are in directory with signal png and pt
                    y, max_len = plot_result(alg_dir, label=algorithm, median=False, cumulative=True)
                    min_max_len = min(min_max_len, max_len)
                plt.legend()
                plt.xlabel("#Iters")
                plt.ylabel("Objective")
                plt.xlim([0, min_max_len])
                plt.yscale("log")
                plt.savefig(os.path.join(exp_dir, "plot_result_regretlogpng.png"), bbox_inches='tight')
                plt.savefig(os.path.join(exp_dir, "plot_result_regretlogpdf.pdf"), bbox_inches='tight')
                plt.clf()
            except:
                continue