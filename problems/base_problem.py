# Base Problem class
from typing import Optional
import networkx as nx
import torch


class Problem:

    def __init__(self,
                 context_graph: nx.Graph,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False):
        self.noise_std = noise_std
        self.negate = negate
        self.log = log
        self.context_graph = context_graph
        self.problem_size = None

    def get_context_adj_mat(self) -> torch.Tensor:
        A = nx.to_numpy_array(self.context_graph)
        A = torch.from_numpy(A)
        return A

    @property
    def is_moo(self) -> bool:
        raise NotImplementedError

    def evaluate_true(self, X):
        raise NotImplementedError

    def __call__(self, X: torch.Tensor, noise: bool = False):
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X).to(dtype=torch.float, device=X.device)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        if self.log:
            f = torch.log(f)
            # f = torch.log(f + 1e-4)
        f += 1e-6 * torch.randn_like(f)
        return f if batch else f.squeeze(0)
