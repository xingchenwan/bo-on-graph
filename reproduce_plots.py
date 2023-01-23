import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from copy import deepcopy

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
    if len(np.unique(n_data_per_trial)) > 1:
        # pad as appropriate
        max_len = max(n_data_per_trial)
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
    plt.plot(x, mean, ".-", label=label, **plot_kwargs)
    if "alpha" in plot_kwargs.keys():
        del plot_kwargs["alpha"]
    if "markevery" in plot_kwargs.keys():
        del plot_kwargs["markevery"]
    plt.fill_between(x, lb, ub, alpha=0.1, **plot_kwargs)
    return y

if __name__ == "__main__":
    d_label = {"ei_ego_network_1":"bo", "random":"random", "local_search":"local_search"}
    logs_dir = './logs'
    task_list = [name for name in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, name))]
    for task in task_list:
        task_dir = os.path.join(logs_dir, task)
        exp_name = [name for name in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, name))]
        for experiment in exp_name:
            exp_dir = os.path.join(task_dir, experiment)
            algorithm_name = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))]
            for algorithm in algorithm_name:
                alg_dir = os.path.join(exp_dir, algorithm)
                ## Here are in directory with signal png and pt
                y = plot_result(alg_dir, label=d_label[algorithm], median=False, cumulative=True)
            plt.legend()
            plt.xlabel("#Iters")
            plt.ylabel("Objective")
            # plt.yscale("log")
            plt.savefig(os.path.join(exp_dir, "plot_result_regretpng.png"))
            plt.savefig(os.path.join(exp_dir, "plot_result_regretpdf.pdf"))
            plt.clf()