import networkx as nx
from typing import Optional, Any, Union, Tuple, Callable

import numpy as np
import torch
from problems.base_problem import Problem
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import random

all_synthetic_problem_labels = [
    "small_ba_betweenness", "diffusion",
]

def get_synthetic_problem(
        label="small_ba_betweenness",
        seed: int = 0
) -> "SyntheticProblem":
    if label == "small_ba_betweenness":
        g = generate_random_graph("ba", seed=seed, return_adj_matrix=False, n=1000, m=3)
        ground_truth = compute_synthetic_node_features(g, feature_name="betweenness",)
        obj_func = lambda idx: ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes))
    elif label == "diffusion":
        g = generate_random_graph("ba", seed=seed, return_adj_matrix=False, n=1000, m=3)
        
        model = ep.SIRModel(g, seed=seed)
        config = mc.Configuration()
        config.add_model_parameter('beta', 0.01)
        config.add_model_parameter('gamma', 0.01)

        config.add_model_parameter("fraction_infected", 0.05)
        model.set_initial_status(config)

        ground_truth = compute_synthetic_node_features(g, feature_name="diffusion", model = model)
        obj_func = lambda idx: ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes))
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
                 log: bool = False):
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

def generate_random_graph(
        graph_type: str = "ba",
        seed: Optional[int] = None,
        return_adj_matrix: bool = False,
        **generator_kwargs,
) -> Union[nx.Graph, Tuple[nx.Graph, np.ndarray]]:
    assert graph_type in ["ba", "ws", "sbm"]
    if graph_type == "ba":
        assert "n" in generator_kwargs.keys()
        assert "m" in generator_kwargs.keys()
        g = nx.generators.random_graphs.barabasi_albert_graph(seed=seed, **generator_kwargs)
    elif graph_type == "sbm":
        assert "sizes" in generator_kwargs.keys()
        assert "p" in generator_kwargs.keys()
        g = nx.stochastic_block_model(seed=seed, **generator_kwargs)
    else:
        raise ValueError(f"Unknown graph_type {graph_type}")
    if not return_adj_matrix:
        return g
    A = nx.adjacency_matrix(g).todense()
    return g, A


def compute_synthetic_node_features(
        input_graph: nx.Graph,
        feature_name: str = "betweenness",
        log: bool = False,
        model: ep.SIRModel = None,
        **kwargs
):
    nnodes = len(input_graph)
    if feature_name == "betweenness":
        feature = nx.betweenness_centrality(input_graph, **kwargs)
    elif feature_name == "eigenvector_centrality":
        feature = nx.eigenvector_centrality(input_graph, **kwargs)
    elif feature_name == "diffusion":
        #initial status
        feature = dict.fromkeys(range(nnodes),0)
        iteration = model.iteration()
        feature = iteration['status']
        while (iteration['node_count'][1] != 0) and (iteration['iteration'] < 50):
            iteration = model.iteration()
            for key, value in iteration['status'].items():
                if value == 1:
                    feature[key] = iteration['iteration'] + 1
        
        for key, value in feature.items():
            if value != 0:
                feature[key] = (1 - (feature[key] - 1)/(iteration['iteration'] + 1))**2

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
    if log:
        return torch.log(
            torch.max(ret, torch.tensor(1e-5).to(ret))
        )
    return ret


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_graph = generate_random_graph("ba", seed=0, n=100, m=2)
    features = compute_synthetic_node_features(
        test_graph,
        "small_ba_betweenness",
        log=True
    )
    dcent_color = [features[i] for i in range(len(features))]
    nx.draw(test_graph, node_color=dcent_color,)
    plt.show()