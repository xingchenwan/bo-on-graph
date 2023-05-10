from search.run_one_replicate import run_one_replication
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import utils.config_utils as config_utils

d_label = {"ei_ego_network_1":"bo", "ei_ego_network_2":"bo_2", "random":"random", "local_search":"local_search", "dfs":"dfs", "bfs":"bfs"}


## TODO Manage option for gpu
if __name__ == "__main__":

    # parse arguments and load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test')
    args = parser.parse_args()
    with open(f'config/{args.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    ### Place where design save_path from config parameters
    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### Get different  configs
    problem_name=config["problem_name"]
    exp_kwargs = config["exp_settings"]
    problem_kwargs = config["problem_settings"]
    bo_kwargs = config["bo_settings"]

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
    for key, value in exp_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["exp", key]))
            list_values.append(value)
    
    if len(list_values) > 0:
        for t in product(*list_values):
            for i, el in enumerate(t):
                type_param, key = list_keys[i]
                if type_param == "pb":
                    problem_kwargs[key] = el
                elif type_param == "bo":
                    bo_kwargs[key] = el
                elif type_param == "exp":
                    exp_kwargs[key] = el

            for seed in range(exp_kwargs["n_exp"]):
                """
                try:
                    algo = bo_kwargs["algorithm"]
                    save_path = os.path.join(save_dir, algo)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    initial_seed = config["seed"]
                    run(save_path=save_path,
                        problem_name=problem_name,
                        seed = initial_seed + seed,
                        exp_kwargs=exp_kwargs,
                        bo_kwargs=bo_kwargs,
                        problem_kwargs=problem_kwargs,
                        )
                except:
                    print(f"Run failed at parameters {t}, proceeding to the next parameters...")
                    continue
                """
                algo = bo_kwargs["algorithm"]
                save_path = os.path.join(save_dir, algo)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                initial_seed = config["seed"]
                run(save_path=save_path,
                    problem_name=problem_name,
                    seed = initial_seed + seed,
                    exp_kwargs=exp_kwargs,
                    bo_kwargs=bo_kwargs,
                    problem_kwargs=problem_kwargs,
                    )
    else:
        for seed in range(exp_kwargs["n_exp"]):
            label = bo_kwargs["algorithm"]
            save_path = os.path.join(save_dir, label)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            initial_seed = config["seed"]
            run(save_path=save_path,
                problem_name=problem_name,
                seed = initial_seed + seed,
                exp_kwargs=exp_kwargs,
                bo_kwargs=bo_kwargs,
                problem_kwargs=problem_kwargs,
                )
    
    plot_figure(config, os.path.dirname(save_path))
    plot_figure(config, os.path.dirname(save_path), log_transform=True)

    ## config["test_function"]
    plot_distribution_gif(config, n_seeds=1)

def main(config):

    seed = 0
    problem_name = config["problem_name"]       # defines the problem
    labels = config["label"]                  # defines the method
    # save_dir = os.path.join(config.save_dir, problem_name)
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_exp = getattr(config, "n_exp", 10)
    plot_result = getattr(config, "plot_result", True)
    animate = getattr(config, "animate", False)
    all_data_over_labels = {l: [] for l in labels}

    assert hasattr(config, "bo_settings")

    for label_idx, label in enumerate(labels):
        all_data = all_data_over_labels[label]
        for i in range(n_exp):
            run_one_replication(
                label=label,
                seed=seed + i,
                problem_name=problem_name,
                save_path=save_dir,
                batch_size=getattr(config.bo_settings, "batch_size", 1),
                n_initial_points=getattr(config.bo_settings, "n_init", 10),
                iterations=getattr(config.bo_settings, "max_iters", 50),
                max_radius=getattr(config.bo_settings, "max_radius", 10),
                context_graph_nnode_init=getattr(
                    config.bo_settings, "context_graph_nnode_init", 100),
                animation=animate,
                trust_region_kwargs=getattr(
                    config.bo_settings, "tr_settings", None),
                problem_kwargs=getattr(config, "problem_settings", None),
            )

    for label_idx, label in enumerate(labels):
        all_data = all_data_over_labels[label]
        for i in range(n_exp):
            load_path = os.path.join(
                save_dir, label, f"{str(i).zfill(4)}_{label}.pt")
            data = torch.load(load_path)["regret"].flatten().numpy()
            all_data.append(data)
        n_data_per_trial = np.array([len(d) for d in all_data])

        # whether the data are rugged
        if len(np.unique(n_data_per_trial)) > 1:
            # pad as appropriate
            max_len = max(n_data_per_trial)
            for i, d in enumerate(all_data):
                all_data[i] = np.concatenate((
                    d, d[-1] * np.ones(max_len - d.shape[0])))
        all_data = np.array(all_data)
        if plot_result:
            x = np.arange(all_data.shape[1])
            mean = pd.DataFrame(all_data).cummin(axis=1).mean(axis=0)
            std = pd.DataFrame(all_data).cummin(axis=1).std(axis=0)
            plt.plot(x, mean, ".-", label=d_label[label], color=f'C{label_idx}')
            plt.fill_between(x, mean - std, mean + std,
                             color=f'C{label_idx}', alpha=0.2)
    if plot_result:
        plt.legend()
        plt.savefig(os.path.join(save_dir, "plot_result.pdf"))
        plt.savefig(os.path.join(save_dir, "plot_result_png.png"))


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='centrality')
    args = parser.parse_args()

    # load parameters
    config = config_utils.setup(f'config/{args.config}.yaml')
    main(config)