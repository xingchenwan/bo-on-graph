from search.run_one_replicate import run_one_replication
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os
import yaml

def main(config):

    save_dir = config["save_dir"]
    seed = 0
    problem_name = config["problem_name"]       # defines the problem
    labels = config["label"]                  # defines the method 
    n_exp = config["n_exp"]
    n_stop = config["plot"]["n_stop"]

    for label in labels:
        for i in range(n_exp):
            run_one_replication(
                label=label,
                seed=seed + i,
                problem_name=problem_name,
                save_path=save_dir,
                batch_size=1,
                n_initial_points=10,
                iterations=50,
                animation=False
            )

        regret_ego = torch.zeros((n_exp, n_stop))
        regret_local = torch.zeros((n_exp, n_stop))
        regret_random = torch.zeros((n_exp, n_stop))

        for i in range(n_exp):

            path_ego = os.path.join(save_dir, "ei_ego_network_1/", f"{str(i).zfill(4)}_ei_ego_network_1.pt")
            ego_load = torch.load(path_ego)["regret"].flatten()
            n_ego = int(ego_load.size()[0])
            if n_ego >= n_stop:
            
                regret_ego[i,:] = ego_load[:n_stop]
            else:
                regret_ego[i,:n_ego] = ego_load[:n_ego]
                regret_ego[i,n_ego:] = ego_load[-1]

            path_local = os.path.join(save_dir, "local_search/", f"{str(i).zfill(4)}_local_search.pt")
            local_load = torch.load(path_local)["regret"].flatten()
            n_local = int(local_load.size()[0])
            if n_local >= n_stop:
            
                regret_local[i,:] = local_load[:n_stop]
            else:
                regret_local[i,:n_local] = local_load[:n_local]
                regret_local[i,n_local:] = local_load[-1]

            path_random = os.path.join(save_dir, "random/", f"{str(i).zfill(4)}_random.pt")
            random_load = torch.load(path_random)["regret"].flatten()
            n_random = int(random_load.size()[0])
            if n_random >= n_stop:
                regret_random[i,:] = random_load[:n_stop]
            else:
                regret_random[i,:n_random] = random_load[:n_random]
                regret_random[i,n_random:] = random_load[-1]
    
    x = range(n_stop)

    mean_random = pd.DataFrame(regret_random.numpy()).cummin(axis = 1).mean(axis=0)
    std_random = pd.DataFrame(regret_random.numpy()).cummin(axis = 1).std(axis=0)**2
    plt.plot(x, pd.DataFrame(regret_random.numpy()).cummin(axis = 1).mean(axis=0), label="random")
    plt.fill_between(x, mean_random - std_random, mean_random + std_random, color='blue', alpha=0.2)

    mean_ego = pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).mean(axis=0)
    std_ego = pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).std(axis=0)**2
    plt.plot(x, pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).mean(axis=0), label="ego")
    plt.fill_between(x, mean_ego - std_ego, mean_ego + std_ego, color='red', alpha=0.2)

    mean_local = pd.DataFrame(regret_local.numpy()).cummin(axis = 1).mean(axis=0)
    std_local = pd.DataFrame(regret_local.numpy()).cummin(axis = 1).std(axis=0)**2
    plt.plot(x, pd.DataFrame(regret_local.numpy()).cummin(axis = 1).mean(axis=0), label="local")
    plt.fill_between(x, mean_local - std_local, mean_local + std_local, color='green', alpha=0.2)
    plt.legend()

    plt.savefig(os.path.join(save_dir, "plot_result.png"))

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1')
    args = parser.parse_args()

    # load parameters
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    main(config)