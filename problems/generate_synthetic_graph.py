from multiprocessing.sharedctypes import Value
from random import random
import networkx as nx
from typing import Optional, Any, Union, Tuple, Callable, Dict

import numpy as np
import torch
from problems.base_problem import Problem
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from math import sqrt


def get_synthetic_problem(
        label: str,
        seed: int = 0,
        n: int = 5000,
        problem_kwargs: Optional[Dict[str, Any]] = None,
) -> "SyntheticProblem":
    problem_kwargs = problem_kwargs or {}
    random_graph_type = problem_kwargs.get("random_graph_type", "ba")
    assert random_graph_type in ["ba", "ws", "grid", "seed", "sbm"]
    if random_graph_type == "ba":
        m = problem_kwargs.get("m", 1)
        g = nx.generators.random_graphs.barabasi_albert_graph(
            seed=seed, n=n, m=m)
    elif random_graph_type == "ws":
        k = problem_kwargs.get("k", 3)
        p = problem_kwargs.get("p", 0.2)
        g = nx.generators.random_graphs.watts_strogatz_graph(
            n=n, k=k, p=p, seed=seed
        )
    elif random_graph_type == "grid":
        n, m = int(sqrt(n)), int(sqrt(n))
        g = nx.generators.grid_2d_graph(n, m)
        mapping = {}
        for i in range(n):
            for j in range(m):
                mapping[(i, j)] = i * m + j
        g = nx.relabel_nodes(g, mapping)
    elif random_graph_type == "sbm":
        sizes = problem_kwargs.get("sizes")
        p = problem_kwargs.get("p")
        g = nx.generators.stochastic_block_model(sizes, p)
    else:
        raise ValueError(
            f"Unknown random_graph_type = {random_graph_type}")

    if label in ["centrality", "small_ba_betweenness"]:  # alias for backward compatibilitys
        feature_name = problem_kwargs.get(
            "feature_name", "eigenvector_centrality")
        ground_truth = compute_synthetic_node_features(
            g, feature_name=feature_name,)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)
    elif label == "diffusion":

        model = ep.SIRModel(g, seed=seed)
        config = mc.Configuration()
        beta = problem_kwargs.get("beta", 1.)
        gamma = problem_kwargs.get("gamma", 0.2)
        fraction_infected = problem_kwargs.get("fraction_infected", 0.0003)
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)

        config.add_model_parameter("fraction_infected", fraction_infected)
        model.set_initial_status(config)

        ground_truth = compute_synthetic_node_features(
            g, feature_name="diffusion", model=model, )

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)

    elif label == "test_function":
        ground_truth = compute_synthetic_node_features(
            g, feature_name="test_function", n=n, m=m, )

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)

    else:
        # todo
        raise NotImplementedError(f"Problem {label} is not implemented")


class SyntheticProblem(Problem):

    is_moo = False

    def __init__(self,
                 context_graph,
                 obj_func: Callable,
                 problem_size: Optional[int] = None,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False,
                 **kwargs,
                 ):
        super().__init__(context_graph, noise_std=noise_std, negate=negate, log=log)
        self.obj_func = obj_func
        self.problem_size = problem_size or len(context_graph)

    @torch.no_grad()
    def evaluate_true(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().long()
        return self.obj_func(X)

    @property
    def ground_truth(self):
        all_possible_vals = torch.arange(self.problem_size).float()
        best = self(all_possible_vals).max()
        return best


def compute_synthetic_node_features(
        input_graph: nx.Graph,
        feature_name: str = "betweenness",
        model: ep.SIRModel = None,
        n=None,
        m=None,
        **kwargs
):
    nnodes = len(input_graph)
    if feature_name == "betweenness":
        feature = nx.betweenness_centrality(input_graph, **kwargs)
    elif feature_name == "eigenvector_centrality":
        feature = nx.eigenvector_centrality(
            input_graph, max_iter=1000, **kwargs)
    elif feature_name == "closeness":
        feature = nx.closeness_centrality(input_graph)
    elif feature_name == "diffusion":
        # initial status
        feature = dict.fromkeys(range(nnodes), 0)
        iteration = model.iteration()
        feature = iteration['status']
        while (iteration['node_count'][1] != 0) and (iteration['iteration'] < 50):
            iteration = model.iteration()
            for key, value in iteration['status'].items():
                if value == 1:
                    feature[key] = iteration['iteration'] + 1

        for key, value in feature.items():
            if value != 0:
                feature[key] = (1 - (feature[key] - 1) /
                                (iteration['iteration'] + 1))**2
    elif feature_name == "test_function":
        feature = dict.fromkeys(range(nnodes), 0)
        def rosembrock(x, y): return -100 * (y - x**2)**2 - (1 - x)**2
        # def rosembrock(x, y): return -y**2 - x**2 + 10
        for i in range(n):
            for j in range(m):
                feature[i * m +
                        j] = rosembrock(2 - (4 / n) * i, -2 + (4 / m) * j)

    else:
        f = getattr(nx, feature_name, None)
        if f:
            feature = f(input_graph, **kwargs)
        else:
            raise ValueError(f"Unknown feature name {feature_name}")
    # the return type is a dict: {node: feature}. Convert them to a tensor
    ret = np.zeros(nnodes)
    for node, val in feature.items():
        ret[node] = val
    ret = torch.from_numpy(ret).float()
    return ret
