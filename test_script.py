# test the GP cross validation
import torch
from search.trust_region import (
    restart,
)
from problems.generate_synthetic_graph import get_synthetic_problem
import matplotlib.pyplot as plt
from search.models import initialize_model
from search.utils import eigendecompose_laplacian
import networkx as nx

def main():
    seed, n, m = 0, 100, 2
    g = nx.generators.random_graphs.barabasi_albert_graph(seed=seed, n=n, m=m)

    model = initialize_model(train_X=torch.zeros(0, dtype=torch.float64),
                            train_Y=torch.zeros(0, dtype=torch.float64),
                            context_graph=g,
                            covar_type="polynomial",
                            covar_kwargs = {"order": 2,})

if __name__ == "__main__":
    main()