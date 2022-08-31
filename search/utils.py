import gpytorch.settings
import networkx as nx
from typing import Union, Tuple, Callable, Optional
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def eigendecompose_laplacian(
        context_graph: nx.Graph,
        dtype: torch.dtype = torch.float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigen-decomposition of ``context_graph``.
        We either take:
            a) a networkx graph.
    Note that the input graphs have to be directed to get a symmetric Laplacian and purely
        real eigenvalues
    Returns a tuple of torch.Tensor of shape ``N`` -> eigenvalues and ``N x N`` eigenvectors
    """
    L = nx.normalized_laplacian_matrix(context_graph).todense()
    L = torch.from_numpy(L).to(dtype)
    eigenvals, eigenvecs = torch.linalg.eigh(L, )
    # eigenvals = eigenvals.real
    # eigenvecs = eigenvecs.real
    # eigenvals = torch.diag(eigenvals)    # eigenvals[1] are the imaginary parts.
    return eigenvals, eigenvecs


def fit_gpytorch_model(mll, model, train_x, train_y,
                       train_iters: int = 100,
                       lr: float = 0.1,
                       print_interval: int = 10,):
    with gpytorch.settings.debug(False):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        model.train()
        model.likelihood.train()
        for i in range(train_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y.squeeze(-1))
            loss.backward()
            if (i + 1) % print_interval == 0:
                print(f"Iter {i+1}/{train_iters}: "
                      f"Loss={loss.item()}. ")
                      # f"Lengthscale={model.covar_module.base_kernel.lengthscale.detach().numpy()}. ")
            optimizer.step()
    return model


def prune_baseline(X_train: torch.Tensor,
                   Y_train: torch.Tensor,
                   index_to_keep: torch.Tensor,
                   ):
    """Remove the training point and targets outside the current context graph"""

    mask = ~(X_train.view(1, -1) != index_to_keep.view(-1, 1)).all(dim=0)
    idx = torch.nonzero(mask).squeeze(1)
    X_train = X_train[idx, ...]
    Y_train = Y_train[idx, ...]
    return X_train, Y_train


def filter_invalid(X: torch.Tensor, X_avoid: torch.Tensor):
    """Remove all occurences of `X_avoid` from `X`."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_avoid.ndim == 1:
        X_avoid = X_avoid.reshape(-1, 1)
    ret = X[~(X == X_avoid.unsqueeze(-2)).all(dim=-1).any(dim=-2)]
    if X.ndim == 1:
        return ret.squeeze(1)
    return ret


def generate_neighbors(
        X: int,
        context_graph:  nx.Graph,
        X_avoid: torch.Tensor,
        stochastic: bool = False
):
    neighbors = torch.tensor(list(nx.all_neighbors(context_graph, int(X)))).to(X_avoid.device)
    valid_neighbors =  filter_invalid(neighbors, X_avoid)
    if stochastic and len(valid_neighbors):
        return random.choice(valid_neighbors).reshape(1, -1)
    return valid_neighbors

# todo: local search for optim acq optimization is not thoroughly tested yet.
def local_search(
        objective_f: Callable,
        context_graph:  Union[nx.Graph, Tuple[torch.Tensor, torch.Tensor]],
        q: int,
        X_avoid: Optional[torch.Tensor] = None,
        stochastic: bool = False,
        num_restarts: int = 20,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        patience = 50,
        unique: bool = True,
        device: str = "cpu",
):
    n_queries = 0
    orig_patience = patience
    candidate_list = []
    if hasattr(objective_f, "X_pending"):
        base_X_pending = objective_f.X_pending if q > 1 else None
    base_X_avoid = X_avoid
    tkwargs = {"device": device, "dtype": torch.int}
    dim = 1 # the input is in terms of the index of graph, so the dimensionality is always 1
    if isinstance(context_graph, nx.Graph):
        nnodes = len(context_graph.nodes)
    else:
        nnodes = context_graph[0].shape[0]
    if X_avoid is None:
        X_avoid = torch.zeros(0, dim, **tkwargs)

    for i in range(q):
        if i == 0 and batch_initial_conditions is not None:
            X0 = filter_invalid(X=batch_initial_conditions, X_avoid=X_avoid)
        else:  # randomly sample nodexs as initial condition
            X0 = torch.randint(nnodes, (num_restarts,)).to(**tkwargs)
            X0 = filter_invalid(X=X0, X_avoid=X_avoid)

        best_xs = torch.zeros(len(X0), dim, **tkwargs)
        best_acqvals = torch.zeros(len(X0), 1, **tkwargs)
        for j, x in enumerate(X0):
            curr_x, curr_f = x.clone(), objective_f(x.unsqueeze(1))
            patience = orig_patience
            while True:
                # find all neighbors on the context graph
                neighbors = generate_neighbors(int(x), context_graph, X_avoid)
                if len(neighbors) == 0:
                    break
                if stochastic:
                    while patience and len(neighbors):
                        with torch.no_grad():
                            acq_val_neighbors = objective_f(neighbors.unsqueeze(1))
                            n_queries += 1
                            if acq_val_neighbors[0] <= curr_f:
                                patience -= 1
                            else:
                                curr_x, curr_f = neighbors, acq_val_neighbors[0]
                                break
                            neighbors = generate_neighbors(int(x), context_graph, X_avoid)
                else:
                    with torch.no_grad():
                        acq_val_neighbors = objective_f(neighbors.unsqueeze(1))
                        n_queries += neighbors.shape[0]
                    if acq_val_neighbors.max() <= curr_f:
                        break   # local minimum reached
                    best_ind = acq_val_neighbors.argmax().item()
                    curr_x, curr_f = neighbors[best_ind].unsqueeze(0), acq_val_neighbors[best_ind]
            best_xs[j, :], best_acqvals[j] = curr_x, curr_f
        # pick the best
        best_idx = best_acqvals.argmax()
        candidate_list.append(best_xs[best_idx].unsqueeze(0))
        # set pending points
        candidates = torch.cat(candidate_list, dim=-2)
        if q > 1:
            if hasattr(objective_f, "set_X_pending"):
                objective_f.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2)
                    if base_X_pending is not None
                    else candidates
                )
            # Update points to avoid if unique is True
            if unique:
                X_avoid = (
                    torch.cat([base_X_avoid, candidates], dim=-2)
                    if base_X_avoid is not None
                    else candidates
                )
    # Reset acq_func to original X_pending state
    if q > 1:
        if hasattr(objective_f, "set_X_pending"):
            objective_f.set_X_pending(base_X_pending)
    with torch.no_grad():
        acq_value = objective_f(candidates)  # compute joint acquisition value
    return candidates, acq_value, n_queries

class Plot_animation:
    def __init__(self, graph, n, ground_truth, save_path = "") -> None:
        self.graph = graph
        self.n = n
        self.ground_truth = ground_truth
        mapping = {}
        for i in range(n):
            for j in range(n):
                mapping[i*n+j] = np.array([1 - i*2/n, -1 + j*2/n])
        
        self.pos = mapping

        self.colors=[float(self.ground_truth(i)) for i in range(n**2)]

        self.save_path = os.path.join("./logs/plot/", save_path)
        self.iteration = 0

    def add_candidate(self, candidates):

        for candidate in candidates:
            self.colors[int(candidate[0])] = -3

    def make_plot(self):

        cmap=plt.cm.Blues
        vmin = min(self.colors)
        vmax = max(self.colors)

        nx.draw(self.graph, self.pos, node_color=self.colors, cmap=cmap, vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        sm._A = []
        #plt.colorbar(sm)
        plt.savefig(os.path.join(self.save_path, f"{str(self.iteration).zfill(4)}.png"))
        self.iteration += 1

    def make_animation(self):
        return 0