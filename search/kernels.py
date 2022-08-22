import gpytorch.kernels
from gpytorch.kernels import Kernel
import torch
from typing import Optional
from search.utils import eigendecompose_laplacian
import networkx as nx

class DiffusionGraphKernel(Kernel):

    has_lengthscale = True

    def __init__(self,
                 context_graph: nx.Graph = None,
                 eigenbasis: Optional[torch.Tensor] = None,
                 eigenvalues: Optional[torch.Tensor] = None,
                 precompute_eigendecompose: bool = True,
                 **kwargs) -> None:
        super(DiffusionGraphKernel, self).__init__(**kwargs)
        # check dimensions
        assert context_graph is not None or (eigenbasis is not None and eigenvalues is not None)

        if eigenvalues is None or eigenbasis is None:
            if precompute_eigendecompose:
                eigenvalues, eigenbasis = eigendecompose_laplacian(context_graph)
            else:
                eigenvalues, eigenbasis = None, None
        else:
            # assert eigenbasis.shape == laplacian.shape
            assert eigenvalues.ndimension() == 1 # and eigenvalues.shape[0] == laplacian.shape[0]
        self.eigenbasis = eigenbasis
        self.eigenvalues = eigenvalues

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError("Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        subvec1 = self.eigenbasis[x1_, ...]   # b1 x ...x bn x n x N
        subvec2 = self.eigenbasis[x2_, ...]   # b1 x ...x bn x m x N
        dists = torch.diag(torch.exp(-self.lengthscale * self.eigenvalues).squeeze())     # N x N
        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2 )
        # res = subvec1 @ torch.diag(dists.squeeze()) @ subvec2.transpose(-1, -2) # b1 x ... x bn x n x m
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    import networkx as nx
    import matplotlib.pyplot as plt
    context_graph = nx.generators.random_graphs.barabasi_albert_graph(seed=0, n=10, m=3)
    label = list(nx.nodes(context_graph))
    labels = dict(zip(label, list(range(len(label)))))
    pos = nx.spring_layout(context_graph)
    nx.draw_networkx_labels(context_graph, pos=pos, labels=labels)
    nx.draw(context_graph, pos=pos)
    plt.show()

    # laplacian = torch.from_numpy(nx.laplacian_matrix(context_graph).todense())
    k = DiffusionGraphKernel(context_graph, )
    k.lengthscale = 0.5
    x1 = torch.arange(len(context_graph))
    K = k(x1, x1).evaluate()
    plt.imshow(K.detach().numpy())
    plt.colorbar()
    plt.show()