from search.run_one_replicate import run_one_replication

save_dir = "./logs/synthetic/"
seed = 0
problem_name = "diffusion"       # defines the problem
label = "ei_ego_network_1"                  # defines the method

labels = ["ei_ego_network_1", "random", "local_search"]

for label in labels:
    for i in range(25):
        run_one_replication(
            label=label,
            seed=i,
            problem_name="diffusion",
            save_path=save_dir,
            batch_size=1,
            n_initial_points=10,
        )