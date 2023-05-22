import argparse
import utils.config_utils as config_utils
from itertools import product
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
matplotlib.rcParams.update({'font.size': 15})

list_color = ["#1f77b4","#8c564b", "#ff7f0e", "#2ca02c","#d62728","#9467bd","#e377c2", "#7f7f7f"]
d_color = {"ei_ego_network_1":"#1f77b4", "ei_ego_network_2":"#8c564b", "random":"#ff7f0e",
           "local_search":"#2ca02c", "dfs": "#d62728", "bfs": "#9467bd", "ei_ego_network_1_old":"#e377c2",
           "ei_ego_network_2_no_ard":"#7f7f7f"}

def plot_result(path: str, label: str, value, index_ablation, plot_kwargs: dict = None, median=False, cumulative=False, regret=True):
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
    #plt.plot(x, mean, ".-", label=label+": " + str(value), color=d_color[label], **plot_kwargs)
    plt.plot(x, mean, ".-", label=label+": " + str(value), color=list_color[index_ablation], **plot_kwargs)
    if "alpha" in plot_kwargs.keys():
        del plot_kwargs["alpha"]
    if "markevery" in plot_kwargs.keys():
        del plot_kwargs["markevery"]
    
    #plt.fill_between(x, lb, ub, alpha=0.1, color=d_color[label], **plot_kwargs)
    plt.fill_between(x, lb, ub, alpha=0.1, color=list_color[index_ablation], **plot_kwargs)
    return y, max_len

### TODO Change path adapted for ablation studies
def create_path(save_path, label, problem_name, ablation_name, problem_kwargs, bo_kwargs, tr_kwargs):
    ### Desiderata:
    # - First name of kernel
    # - Second name of the task
    # - Third name of the graph
    # - Fourth different parameters of kernel, task, graph
    # => Results ordered more conveniently
    if problem_name == "diffusion":
        if problem_kwargs["random_graph_type"] == "ba":
            s = "_".join([problem_kwargs["random_graph_type"], f'm-{problem_kwargs["m"]}', f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}',
                f'n-{problem_kwargs["n"]}', f'epsilon-{problem_kwargs["epsilon"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}', f'iter-{problem_kwargs["iteration_diffusion"]}'
                ])
        elif problem_kwargs["random_graph_type"] == "ws":
            s = "_".join([problem_kwargs["random_graph_type"], f'k-{problem_kwargs["k"]}', f'p-{problem_kwargs["p"]}', f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}',
                f'n-{problem_kwargs["n"]}', f'epsilon-{problem_kwargs["epsilon"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}', f'iter-{problem_kwargs["iteration_diffusion"]}'
                ])
    elif problem_name == "diffusion_real":
        s = "_".join([f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}'])
    elif problem_name == "centrality":
        if problem_kwargs["random_graph_type"] == "ba":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'm-{problem_kwargs["m"]}', f'ninit-{problem_kwargs["n_init"]}', f'n-{problem_kwargs["n"]}'])
        if problem_kwargs["random_graph_type"] == "ws":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'k-{problem_kwargs["k"]}', f'p-{problem_kwargs["p"]}', f'ninit-{problem_kwargs["n_init"]}', f'n-{problem_kwargs["n"]}'])
        if problem_kwargs["random_graph_type"] == "real_enron":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
        if problem_kwargs["random_graph_type"] == "real_twitch":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
        if problem_kwargs["random_graph_type"] == "real_fb":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
    elif problem_name == "test_function":
        s = "_".join([problem_kwargs["test_function"], f'noise-{problem_kwargs["noise"]}', f'n-{problem_kwargs["n"]}'])
    elif problem_name == "team_opt":
        s = "_".join([problem_kwargs["random_graph_type"], f'n_skills-{problem_kwargs["n_skills"]}', f'alpha-{problem_kwargs["alpha"]}', f'n_individuals-{problem_kwargs["n_individuals"]}', f'threshold-{problem_kwargs["threshold"]}'])
    
    ##str_bo = [key + "-" + f"{value}" for key, value in bo_kwargs.items() if key not in ["tr_settings", ablation_name]]
    ##str_tr = [key + "-" + f"{value}" for key, value in tr_kwargs.items() if key != ablation_name]
    str_bo = [key + "-" + f"{value}" for key, value in bo_kwargs.items() if key not in ["tr_settings"]]
    str_tr = [key + "-" + f"{value}" for key, value in tr_kwargs.items()]
    s = "_".join([label, problem_name, s] + str_bo + str_tr)
    save_path = os.path.join(save_dir, s)
    return save_path


#d_label = {"ei_ego_network_1":"bo", "ei_ego_network_2":"bo_2", "random":"random", "local_search":"local_search", "dfs":"dfs", "bfs":"bfs"}

## TODO Manage option for gpu
if __name__ == "__main__":
    OVERWRITE = True
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='centrality')
    args = parser.parse_args()

    # load parameters
    config = config_utils.setup(f'config/{args.config}.yaml')
    
    ### Place where design save_path from config parameters
    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ### Get different  configs
    labels = config["label"]                  # defines the method
    problem_name=config["problem_name"]
    ablation_name=config["problem_ablation"]
    problem_kwargs = config["problem_settings"]
    bo_kwargs = config["bo_settings"]
    tr_kwargs = bo_kwargs["tr_settings"]
    n_exp = getattr(config, "n_exp", 10)
    plot_result = getattr(config, "plot_result", True)
    animate = getattr(config, "animate", False)
    all_data_over_labels = {l: [] for l in labels}
    seed=0

    ### Check path
    save_dir_ablation = "./logs/ablation/ablation_name/" + args.config + "/"
    if not os.path.exists(save_dir_ablation):
        os.makedirs(save_dir_ablation)

    ## Get list of parameters for the ablation study
    if ablation_name in ["max_radius", "context_graph_nnode_init"]:
        list_ablation_parameter = bo_kwargs[ablation_name]
    elif ablation_name in ["n_nodes_min", "trust_region_multiplier", "succ_tol", "fail_tol"]:
        list_ablation_parameter = tr_kwargs[ablation_name]

    ### Make lists for multiple experiments ### Exchange label and ablation parameter here
    list_keys, list_values = [], []
    for key, value in problem_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["pb", key]))
            list_values.append(value)
    for key, value in bo_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["bo", key]))
            list_values.append(value)
    for key, value in tr_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["tr", key]))
            list_values.append(value)
    if len(list_values) > 0:
        for label in labels:
            for index_ablation, t in enumerate(product(*list_values)):
                for i, el in enumerate(t):
                    type_param, key = list_keys[i]
                    if type_param == "pb":
                        problem_kwargs[key] = el
                    elif type_param == "bo":
                        bo_kwargs[key] = el
                    elif type_param == "tr":
                        tr_kwargs[key] = el

                save_path = create_path(save_dir, label, problem_name, ablation_name, problem_kwargs, bo_kwargs, tr_kwargs)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print("Processing", save_path, "...")
                else:
                    "If folder already exists then perform optimization depending on OVERWRITE"
                    if OVERWRITE == False:
                        print(save_path + "found without overwriting, next config...")
                        continue
                bo_kwargs["tr_settings"] = tr_kwargs
                ### Here process plots from n_seeds and save it in the path save_dir_ablation_kernel
                ## Assumption that only ablation parameter is under list form
                y, max_len = plot_result(save_path, label, value=t[0], index_ablation=index_ablation, median=False, cumulative=True)
                min_max_len = min(min_max_len, max_len)
            save_dir_ablation_kernel = os.path.join(save_dir_ablation, label)
            plt.legend()
            plt.xlabel("#Iters")
            plt.ylabel("Objective")
            plt.xlim([0, min_max_len])
            plt.savefig(os.path.join(save_dir_ablation_kernel, "plot_result_regretpng.png"), bbox_inches='tight')
            plt.savefig(os.path.join(save_dir_ablation_kernel, "plot_result_regretpdf.pdf"), bbox_inches='tight')
            plt.yscale("log")
            plt.savefig(os.path.join(save_dir_ablation_kernel, "plot_result_regretlogpng.png"), bbox_inches='tight')
            plt.savefig(os.path.join(save_dir_ablation_kernel, "plot_result_regretlogpdf.pdf"), bbox_inches='tight')
            plt.clf()

    else:
        raise NotImplementedError