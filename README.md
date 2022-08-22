# Bayesian optimization on graphs


## Create virtual env & install dependencies
```
conda create -n graph
conda install networkx numpy pandas matplotlib seaborn jupyterlab
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda activate graph
```

## Run
Use the following code block to run a single trial:
```python

from search.run_one_replicate import run_one_replication

save_dir = "./logs/synthetic/"
seed = 0
problem_name = "small_ba_betweenness"       # defines the problem
label = "ei_ego_network_1"                  # defines the method

run_one_replication(
    label,
    seed=seed,
    problem_name=problem_name,
    save_path=save_dir,
    batch_size=1,
    n_initial_points=10,
)
```
