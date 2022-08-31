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
    labels = ["local_search"]    

    for label in labels:
        for i in range(1):
            run_one_replication(
                label=label,
                seed=seed + i,
                problem_name="diffusion",
                save_path=save_dir,
                batch_size=1,
                n_initial_points=10,
                iterations=50,
                animation=True
            )

    regret_ego = torch.zeros((10, 60))
    regret_local = torch.zeros((10, 60))
    regret_random = torch.zeros((10, 60))

    for i in range(10):
        path_ego = os.path.join(save_dir, "ei_ego_network_1/", f"{str(i).zfill(4)}_ei_ego_network_1.pt")
        regret_ego[i,:] = torch.load(path_ego)["regret"].flatten()[:60]

        path_local = os.path.join(save_dir, "local_search/", f"{str(i).zfill(4)}_local_search.pt")
        regret_local[i,:] = torch.load(path_local)["regret"].flatten()[:60]

        path_ego = os.path.join(save_dir, "random/", f"{str(i).zfill(4)}_random.pt")
        regret_random[i,:] = torch.load(path_ego)["regret"].flatten()[:60]
    
    x = range(60)

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    main(args.name)