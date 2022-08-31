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
    labels = ["random", "local_search"]    

    for label in labels:
        run_one_replication(
            label=label,
            seed=seed,
            problem_name="test_function",
            save_path=save_dir,
            batch_size=1,
            n_initial_points=20,
            iterations=100,
            animation=True
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    main(args.name)