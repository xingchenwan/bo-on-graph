import networkx as nx
from typing import Optional, Any, Union, Tuple

import numpy as np


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
):
    pass
