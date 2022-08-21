from gpytorch.kernels import Kernel
import torch
from typing import Optional
from .utils import eigendecompose_laplacian


class DiffusionGraphKernel(Kernel):
    
    def __init__(self,
                 laplacian: torch.Tensor = None,
                 eigenbasis: Optional[torch.Tensor] = None,
                 eigenvalues: Optional[torch.Tensor] = None,
                 precompute_eigendecompose: bool = True,
                 **kwargs) -> None:
        super(DiffusionGraphKernel, self).__init__(**kwargs)
        # check dimensions
        assert laplacian is not None or (eigenbasis is not None and eigenvalues is not None)
        if laplacian is not None:
            assert laplacian.shape[0] == laplacian.shape[1]
        if eigenvalues is None or eigenbasis is None:
            if precompute_eigendecompose:
                eigenvalues, eigenbasis = eigendecompose_laplacian(laplacian)
            else:
                eigenvalues, eigenbasis = None, None
        else:
            assert eigenbasis.shape == laplacian.shape
            assert eigenvalues.ndimension() == 1 and eigenvalues.shape[0] == laplacian.shape[0]
        self.eigenbasis = eigenbasis
        self.eigenvalues = eigenvalues

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        x1: torch.Tensor of shape (n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError("Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.cpu().detach().tolist()
        x2_ = x2.cpu().detach().tolist()
        subvec1 = self.eigenbasis[:, x1_]   # N x n
        subvec2 = self.eigenbasis[:, x2_]   # N x m
        dists = torch.diag(torch.exp(-self.lengthscale * self.eigenvalues)) # N x N
        res = subvec1.transpose(-1, -2) @ dists @ subvec2
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res