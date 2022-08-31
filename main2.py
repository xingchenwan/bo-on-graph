from search.run_one_replicate import run_one_replication
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import os

def main(name):
    save_dir = "./logs/synthetic/" + name
    seed = 0
    problem_name = "diffusion"       # defines the problem
    label = "ei_ego_network_1"                  # defines the method

    labels = ["ei_ego_network_1", "random", "local_search"]

    n_stop = 100

    regret_ego = torch.zeros((10, n_stop))
    regret_local = torch.zeros((10, n_stop))
    regret_random = torch.zeros((10, n_stop))

    for i in range(10):

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

    plt.savefig(os.path.join(save_dir, "plot_resultfull.png"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    main(args.name)