# [NeurIPS 2023] Bayesian Optimisation of Functions on Graphs

Code repository for NeurIPS 2023 paper [Bayesian Optimisation of Functions on Graphs](https://openreview.net/forum?id=UuNd9A6noD).

If you find the paper or the codebase useful to your research, please cite:
```
@inproceedings{
wan2023bayesian,
title={Bayesian Optimisation of Functions on Graphs},
author={Xingchen Wan and Pierre Osselin and Henry Kenlay and Binxin Ru and Michael A Osborne and Xiaowen Dong},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UuNd9A6noD}
}
```

## Create virtual env & install dependencies
```
conda create -n graph
conda install networkx numpy pandas matplotlib seaborn jupyterlab
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda activate graph
pip install ndlib
```

## Run
Use the following code block to run a single trial:
```python

from search.run_one_replicate import run_one_replication

save_dir = "./logs/synthetic/"
seed = 0
problem_name = "small_ba_betweenness"       # defines the problem
label = "ei_ego_network_polynomial"                  # defines the method

run_one_replication(
    label,
    seed=seed,
    problem_name=problem_name,
    save_path=save_dir,
    batch_size=1,
    n_initial_points=10,
)
```
To run through command line, define a configuration in the folder config/config.yaml and run the command:
```
python main.py --config config
```
For example, to run the centrality experiment with BA graphs, run:
```
python main.py --config centrality_ba
```
