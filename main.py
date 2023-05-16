from search.run_one_replicate import run_one_replication
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import utils.config_utils as config_utils
import yaml
from itertools import product

def create_path(save_path, problem_name, problem_kwargs, bo_kwargs):
    
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
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'contextnodes-{bo_kwargs["context_graph_nnode_init"]}', f'n_init-{problem_kwargs["n_init"]}'])
        
    elif problem_name == "test_function":
        s = "_".join([problem_kwargs["test_function"], f'noise-{problem_kwargs["noise"]}', f'n-{problem_kwargs["n"]}'])
    elif problem_name == "team_opt":
        s = "_".join([problem_kwargs["random_graph_type"], f'n_skills-{problem_kwargs["n_skills"]}', f'alpha-{problem_kwargs["alpha"]}', f'n_individuals-{problem_kwargs["n_individuals"]}', f'threshold-{problem_kwargs["threshold"]}'])
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
    problem_kwargs = config["problem_settings"]
    bo_kwargs = config["bo_settings"]
    n_exp = getattr(config, "n_exp", 10)
    plot_result = getattr(config, "plot_result", True)
    animate = getattr(config, "animate", False)
    all_data_over_labels = {l: [] for l in labels}
    seed=10

    ### Make lists for multiple experiments
    list_keys, list_values = [], []
    for key, value in problem_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["pb", key]))
            list_values.append(value)
    for key, value in bo_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["bo", key]))
            list_values.append(value)
    
    if len(list_values) > 0:
        for t in product(*list_values):
            for i, el in enumerate(t):
                type_param, key = list_keys[i]
                if type_param == "pb":
                    problem_kwargs[key] = el
                elif type_param == "bo":
                    bo_kwargs[key] = el

            #### Build new save dir ba_m-3_beta-0.01_gamma-0.005_n-5000_epsilon-5e-4_iter-100_abs
            save_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print("Processing", save_path, "...")
            else:
                "If folder already exists then perform optimization depending on OVERWRITE"
                if OVERWRITE == False:
                    print(save_path + "found without overwriting, next config...")
                    continue

            for label_idx, label in enumerate(labels):
                all_data = all_data_over_labels[label]
                for i in range(n_exp):
                    try:
                        run_one_replication(
                                label=label,
                                seed=seed + i,
                                problem_name=problem_name,
                                save_path=save_path,
                                batch_size=getattr(bo_kwargs, "batch_size", 1),
                                n_initial_points=getattr(bo_kwargs, "n_init", 10),
                                iterations=getattr(bo_kwargs, "max_iters", 50),
                                max_radius=getattr(bo_kwargs, "max_radius", 10),
                                context_graph_nnode_init=getattr(
                                    bo_kwargs, "context_graph_nnode_init", 100),
                                animation=animate,
                                trust_region_kwargs=getattr(
                                    bo_kwargs, "tr_settings", None),
                                problem_kwargs=problem_kwargs,
                            )
                    except Exception as e:
                        print("Configuration with label " + label + "failed, with error " + str(e) + "continue...")
                        continue

            # for label_idx, label in enumerate(labels):
            #     all_data = all_data_over_labels[label]
            #     for i in range(n_exp):
            #         load_path = os.path.join(
            #             save_path, label, f"{str(i).zfill(4)}_{label}.pt")
            #         data = torch.load(load_path)["regret"].flatten().numpy()
            #         all_data.append(data)
            #     n_data_per_trial = np.array([len(d) for d in all_data])

            #     # whether the data are rugged
            #     if len(np.unique(n_data_per_trial)) > 1:
            #         # pad as appropriate
            #         max_len = max(n_data_per_trial)
            #         for i, d in enumerate(all_data):
            #             all_data[i] = np.concatenate((
            #                 d, d[-1] * np.ones(max_len - d.shape[0])))
            #     all_data = np.array(all_data)
            #     if plot_result:
            #         x = np.arange(all_data.shape[1])
            #         mean = pd.DataFrame(all_data).cummin(axis=1).mean(axis=0)
            #         std = pd.DataFrame(all_data).cummin(axis=1).std(axis=0)
            #         plt.plot(x, mean, ".-", label=d_label[label], color=f'C{label_idx}')
            #         plt.fill_between(x, mean - std, mean + std,
            #                             color=f'C{label_idx}', alpha=0.2)
            # if plot_result:
            #     plt.legend()
            #     plt.savefig(os.path.join(save_path, "plot_result.pdf"))
            #     plt.savefig(os.path.join(save_path, "plot_result_png.png"))
            #     plt.clf()


    else:
        save_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
        for label_idx, label in enumerate(labels):
            all_data = all_data_over_labels[label]
            for i in range(n_exp):
                try:
                    run_one_replication(
                        label=label,
                        seed=seed + i,
                        problem_name=problem_name,
                        save_path=save_path,
                        batch_size=getattr(bo_kwargs, "batch_size", 1),
                        n_initial_points=getattr(bo_kwargs, "n_init", 10),
                        iterations=getattr(bo_kwargs, "max_iters", 50),
                        max_radius=getattr(bo_kwargs, "max_radius", 10),
                        context_graph_nnode_init=getattr(
                            bo_kwargs, "context_graph_nnode_init", 100),
                        animation=animate,
                        trust_region_kwargs=getattr(
                            bo_kwargs, "tr_settings", None),
                        problem_kwargs=problem_kwargs,
                    )
                except Exception as e:
                    print("Configuration with label " + label + "failed, with error " + str(e) + "continue...")
                    continue
    
        # for label_idx, label in enumerate(labels):
        #     all_data = all_data_over_labels[label]
        #     for i in range(n_exp):
        #         load_path = os.path.join(
        #             save_path, label, f"{str(i).zfill(4)}_{label}.pt")
        #         data = torch.load(load_path)["regret"].flatten().numpy()
        #         all_data.append(data)
        #     n_data_per_trial = np.array([len(d) for d in all_data])

        #     # whether the data are rugged
        #     if len(np.unique(n_data_per_trial)) > 1:
        #         # pad as appropriate
        #         max_len = max(n_data_per_trial)
        #         for i, d in enumerate(all_data):
        #             all_data[i] = np.concatenate((
        #                 d, d[-1] * np.ones(max_len - d.shape[0])))
        #     all_data = np.array(all_data)
        #     if plot_result:
        #         x = np.arange(all_data.shape[1])
        #         mean = pd.DataFrame(all_data).cummin(axis=1).mean(axis=0)
        #         std = pd.DataFrame(all_data).cummin(axis=1).std(axis=0)
        #         plt.plot(x, mean, ".-", label=d_label[label], color=f'C{label_idx}')
        #         plt.fill_between(x, mean - std, mean + std,
        #                             color=f'C{label_idx}', alpha=0.2)
        # if plot_result:
        #     plt.legend()
        #     plt.savefig(os.path.join(save_path, "plot_result.pdf"))
        #     plt.savefig(os.path.join(save_path, "plot_result_png.png"))
        #     plt.clf()