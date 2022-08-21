import networkx as nx
from typing import Union, Tuple
import torch


def eigendecompose_laplacian(
        input_graph: Union[nx.Graph, Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigen-decomposition of ``input_graph``.
        We either take:
            a) a networkx graph.
            b) a tuple of {adjacency matrix, degree matrix} of shape ``N x N`` each.
    Note that the input graphs have to be directed to get a symmetric Laplacian and purely
        real eigenvalues
    Returns a tuple of torch.Tensor of shape ``N`` -> eigenvalues and ``N x N`` eigenvectors
    """
    if isinstance(input_graph, nx.Graph):
        L = nx.laplacian_matrix(input_graph).todense()
        L = torch.from_numpy(L)
    else:
        assert input_graph[0].shape == input_graph[1].shape
        L = input_graph[1] - input_graph[0]
    eigenvals, eigenvecs = torch.linalg.eig(L, eigenvectors=True)
    eigenvals = torch.diag(eigenvals[0])    # eigenvals[1] are the imaginary parts.
    return eigenvals, eigenvecs