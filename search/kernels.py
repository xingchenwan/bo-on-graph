from turtle import forward
import gpytorch.kernels
from gpytorch.kernels import Kernel
import torch
from typing import Optional
#from search.utils import eigendecompose_laplacian
from .utils import eigendecompose_laplacian
import networkx as nx


class DiffusionGraphKernel(Kernel):

    has_lengthscale = True

    def __init__(self,
                 context_graph: nx.Graph = None,
                 eigenbasis: Optional[torch.Tensor] = None,
                 eigenvalues: Optional[torch.Tensor] = None,
                 precompute_eigendecompose: bool = True,
                 normalized_laplacian: bool = True,
                 order: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # check dimensions
        assert context_graph is not None or (
            eigenbasis is not None and eigenvalues is not None)

        if eigenvalues is None or eigenbasis is None:
            if precompute_eigendecompose:
                eigenvalues, eigenbasis = eigendecompose_laplacian(
                    context_graph,
                    normalized_laplacian=normalized_laplacian)
            else:
                eigenvalues, eigenbasis = None, None
        else:
            assert eigenvalues.ndimension() == 1
        self.eigenbasis = eigenbasis
        self.eigenvalues = eigenvalues
        self.order = order

    def get_dist(self):
        order = min(
            self.order, self.eigenvalues.shape[0]) if self.order else self.eigenvalues.shape[0]
        effective_eigenvalues = self.eigenvalues[:order]

        dists = (
            torch.exp(-effective_eigenvalues *
                      self.lengthscale))

        if order > 1:
            dists = torch.diag(dists.squeeze())
            dists *= order / torch.sum(dists)
        # else:
        #     dists = dists.unsqueeze(0)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError(
                "Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        order = min(
            self.order, self.eigenvalues.shape[0]) if self.order else self.eigenvalues.shape[0]

        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, :order]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, :order]
        dists = self.get_dist()     # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res

class PolynomialKernel(DiffusionGraphKernel):

    has_lengthscale = True

    def get_dist(self):  
        epsilon = 1e-6
        # Note the definition of the B matrix here -- we directly power the eigenvalues
        # without the inversion in the previous iteration.
        eigen_powered = torch.cat(
            [(self.eigenvalues ** i).reshape(1, -1) for i in range(self.order)]
        )  # shape: (self.order, n)
        # This is the B matrix
        #dists = torch.einsum("ij,i->ij", eigen_powered,self.beta.squeeze(0))
        # Sum B matrix
        dists = torch.einsum("ij,i->ij", eigen_powered, self.lengthscale.squeeze(0))
        dists = torch.diag(1/(dists.sum(0).squeeze() + epsilon))
        # print(dists, self.beta)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError(
                "Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, ...]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, ...]
        dists = self.get_dist()     # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


class PolynomialKernelOld(DiffusionGraphKernel):

    has_lengthscale = True

    def get_dist(self):
        epsilon = 1e-6
        eigen_powered = torch.stack(
            [self.eigenvalues ** i for i in range(self.order)]
        )

        dists = 1. / (torch.einsum("ij,i->ij", eigen_powered,
                                   self.lengthscale.squeeze(0)) + epsilon)
        dists = torch.diag(dists.sum(0).squeeze())
        dists *= self.eigenvalues.shape[0] / torch.sum(dists)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError(
                "Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, ...]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, ...]
        dists = self.get_dist()     # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res
    
class MaternKernel(DiffusionGraphKernel):
    has_lengthscale = True
    def __init__(self,
                 nu: Optional[float] = 2.5,
                 context_graph: nx.Graph = None,
                 eigenbasis: Optional[torch.Tensor] = None,
                 eigenvalues: Optional[torch.Tensor] = None,
                 precompute_eigendecompose: bool = True,
                 normalized_laplacian: bool = True,
                 order: int = None,
                 **kwargs) -> None:
        super(MaternKernel, self).__init__(context_graph,
                                           eigenbasis,
                                           eigenvalues,
                                           precompute_eigendecompose,
                                           normalized_laplacian,
                                           order,
                                           **kwargs)
        self.nu = nu
    
    def get_dist(self):
        dists = (self.eigenvalues + (2*self.nu)/(self.lengthscale**2))**(-self.nu)
        return torch.diag(dists.squeeze(0))

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: torch.Tensor of shape (b1 x ... x bn x n x 1): each element is a vertice index.
        x2: torch.Tensor of shape (b1 x ... x bn x m x 1): each element is a vertice index.
        Output:
            kernel matrix of dim (b1 x .... x bn x n x m)
        Note that this kernel is not differentiable w.r.t. the inputs.
        """
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError(
                "Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, ...]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, ...]
        dists = self.get_dist()     # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


class PolynomialKernelNew(DiffusionGraphKernel):
    """
    Implementation of the new kernel -- it has the same mathematical form as the previous
    polynomial kernel, but the non-negative constraints are on the sum of eigenvalues
    and the learned parameters (rather than individually). This allows non-low-pass
    behaviour to be learnt.
    """
    # Note that as the \betas now can be negative and cannot
    # be interpreted as lengthscales.

    has_lengthscale = True

    def __init__(self, beta_prior=None, **kwargs):
        super().__init__(**kwargs)


        default_beta_prior = gpytorch.priors.NormalPrior(
            torch.zeros(self.order), 1e-1*torch.ones(self.order))
        beta_prior = beta_prior if beta_prior is not None else default_beta_prior
        # Note that we require the betas to be initialized as non-negative (but
        # individual betas are allowed to go negative during optimization), so
        # we set the prior on the sqrt of beta.
        self.register_prior(
            "beta_prior", beta_prior, lambda m: m.beta
        )
        # Set random initialization by sampling from the prior
        init_beta = torch.abs(beta_prior.sample())
        self.register_parameter(
            name="beta", parameter=torch.nn.Parameter(init_beta)
        )

    def get_dist(self):
        # Note the definition of the B matrix here -- we directly power the eigenvalues
        # without the inversion in the previous iteration.
        eigen_powered = torch.cat(
            [(self.eigenvalues ** i).reshape(1, -1) for i in range(self.order)]
        )  # shape: (self.order, n)
        # This is the B matrix
        dists = torch.einsum("ij,i->ij", eigen_powered,
                             self.beta.squeeze(0))
        # Sum B matrix
        dists = torch.diag(dists.sum(0).squeeze())
        dists *= self.eigenvalues.shape[0] / torch.sum(dists)
        # print(dists, self.beta)
        return dists

    def forward(self, x1, x2, diag=False, **params):
        if self.eigenvalues is None or self.eigenbasis is None:
            raise ValueError(
                "Eigendecomposition of Laplacian is not performed!")
        assert x1.shape[-1] == 1 and x2.shape[-1] == 1
        x1_ = x1.long().squeeze(-1)
        x2_ = x2.long().squeeze(-1)
        # b1 x ...x bn x n x N
        subvec1 = self.eigenbasis[x1_, ...]
        # b1 x ...x bn x m x N
        subvec2 = self.eigenbasis[x2_, ...]
        dists = self.get_dist()     # N x N
        self._dists = torch.diagonal(dists.clone(), 0)

        tmp = torch.einsum("...ij,jj->...ij", subvec1, dists)
        res = torch.einsum("...ij,...kj->...ik", tmp, subvec2)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    import networkx as nx
    import matplotlib.pyplot as plt
    context_graph = nx.generators.random_graphs.barabasi_albert_graph(
        seed=0, n=10, m=3)
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
