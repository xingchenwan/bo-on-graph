from search.run_one_replicate import run_one_replication
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import multiprocessing as mp
import utils.config_utils as config_utils

d_label = {"ei_ego_network_1":"bo", "random":"random", "local_search":"local_search", "dfs":"dfs", "bfs":"bfs"}

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
                covar_type=getattr(
                    config.bo_settings, "covar_type", "polynomial"), #diffusion also possible
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
    config = config_utils.setup(f'config/ackley.yaml')


    ## Setting list of configurations


    #label:  ["ei_ego_network_1", "random", "local_search", "dfs", "bfs"]
    #save_dir: ./logs/testfunction/ackley_noise-1/
    #problem_name: test_function

    #problem_settings:
    #    n: 5000
    #    random_graph_type: "grid" # sbm, ba, ws
    #    test_function: "ackley"
    #    noise: 0.
    """
    size_buffer_list = [50, 100]
    N_list = [2, 5, 10]
    T_list = [100, 500, 1000, 10000]
    threshold_list = [0.1, 0.5, 0.7, 1.]
    seed_list = [0, 1, 2]

    paramlist = list(itertools.product(size_buffer_list, N_list, T_list, threshold_list, seed_list))
    list_dict = []

    for buffer, N, T, threshold, seed in paramlist:
        list_dict.append({"size_buffer": buffer, "N": N, "T": T, "threshold": threshold, "seed": seed})
    
    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(sim, list_dict)
        
    for buffer, N, T, threshold, seed in paramlist:
        list_dict.append({"size_buffer": buffer, "N": N, "T": T, "threshold": threshold, "seed": seed})
    
    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(main, list_dict)
    """

  
    main(config)