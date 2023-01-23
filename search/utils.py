
import gpytorch.settings
import networkx as nx
from typing import Union, Tuple, Callable, Optional
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from botorch.cross_validation import gen_loo_cv_folds, CVFolds


def eigendecompose_laplacian(
        context_graph: nx.Graph,
        dtype: torch.dtype = torch.float,
        normalized_laplacian: bool = True,
        normalized_eigenvalues: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigen-decomposition of ``context_graph``.
        We either take:
            a) a networkx graph.
    Note that the input graphs have to be directed to get a symmetric Laplacian and purely
        real eigenvalues
    Returns a tuple of torch.Tensor of shape ``N`` -> eigenvalues and ``N x N`` eigenvectors
    """
    if normalized_laplacian:
        L = nx.normalized_laplacian_matrix(context_graph).todense()
        if normalized_eigenvalues:
            # eigenvalues of normalized Laplacian are bounded by [0, 2].
            # divide by 2 to ensure the eigenvalues are between [0, 1]
            L /= 2.
            # L /= 1.
    else:
        L = nx.laplacian_matrix(context_graph).todense()
    L = torch.from_numpy(L).to(dtype)
    eigenvals, eigenvecs = torch.linalg.eigh(L, )
    return eigenvals, eigenvecs


def fit_gpytorch_model(mll, model, train_x, train_y,
                       train_iters: int = 100,
                       lr: float = 0.1,
                       print_interval: int = -1,
                       return_loss: bool = False):
    with gpytorch.settings.debug(False):
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        model.likelihood.train()
        for i in range(train_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, model.train_targets)
            if loss.ndim > 0:
                loss = loss.sum()
            loss.backward()
            if print_interval > 0 and (i + 1) % print_interval == 0:
                print(f"Iter {i+1}/{train_iters}: "
                      f"Loss={loss.item()}. ")
            optimizer.step()

            # lengthscale = model.covar_module.base_kernel.lengthscale
            # lengthscale[..., :5] = 1.
            # lengthscale[..., 1:] = 1e4  # or whatever value you care about
            # model.covar_module.base_kernel.lengthscale = lengthscale
    if return_loss:
        return model, loss.item()
    return model


def prune_baseline(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    index_to_keep: torch.Tensor,
):
    """Remove the training point and targets outside the current context graph"""

    mask = ~(X_train.view(1, -1) != index_to_keep.view(-1, 1)).all(dim=0)
    idx = torch.nonzero(mask).squeeze(1)
    X_train = X_train[idx, ...]
    Y_train = Y_train[idx, ...]
    return X_train, Y_train


def get_context_graph(
        base_graph: nx.Graph,
        centre_node_idx: int,
        nnodes: Optional[int] = 100,
        n_hop_max: Optional[int] = None,
) -> nx.Graph:
    """
    Obtain the context graph.
    Args:
        base_graph: nx.Graph. a base networkx graph
        centre_node_idx: int. The index of the node of `base_graph` that should be used
            as the centre of the graph construction.
        max_node: the maximum number of nodes of the context graph.
        n_hop: int. The maximum hop distance. This can be overriden by `max_node` argument.
    Returns:
        context_graph: a nx.Graph that is a subgraph of `base_graph`.
    """
    n_hop_max = n_hop_max or float("inf")
    if ((not nnodes) and (not n_hop_max)) or (nnodes >= len(base_graph)):
        return base_graph.copy()
    elif nnodes:
        selected_nodes = set([centre_node_idx])
        # start at the specified central node
        nbrs = set([centre_node_idx])
        current_rad = 0
        while len(selected_nodes) < nnodes and current_rad < n_hop_max:
            nbrs = set((nbr for n in nbrs for nbr in base_graph[n]))
            # remove the nodes that are already selected
            nbrs = nbrs - nbrs.intersection(selected_nodes)
            if len(selected_nodes) + len(nbrs) < nnodes:
                selected_nodes = selected_nodes.union(nbrs)
                # selected_nodes += nbrs
            else:
                # subsample
                nbrs_subsampled_idx = np.random.choice(
                    len(nbrs), nnodes - len(selected_nodes), replace=False).tolist()
                nbrs_subsampled = set([list(nbrs)[i]
                                      for i in nbrs_subsampled_idx])
                # selected_nodes += nbrs_subsampled
                selected_nodes = selected_nodes.union(nbrs_subsampled)
            # denote that we are now sampling `k+1`-hop neighbors
            current_rad += 1
        # get the induced subgraph of the set of nodes in selected_nodes
        context_graph = base_graph.subgraph(list(selected_nodes)).copy()
    elif n_hop_max:
        context_graph = nx.ego_graph(
            base_graph, centre_node_idx, radius=n_hop_max
        )

    return context_graph


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
        context_graph: nx.Graph,
        X_avoid: torch.Tensor,
        stochastic: bool = False
):
    neighbors = torch.tensor(list(nx.all_neighbors(
        context_graph, int(X)))).to(X_avoid.device)
    valid_neighbors = filter_invalid(neighbors, X_avoid)
    if stochastic and len(valid_neighbors):
        return random.choice(valid_neighbors).reshape(1, -1)
    return valid_neighbors

# todo: local search for optim acq optimization is not thoroughly tested yet.

def local_search(
        objective_f: Callable,
        context_graph: Union[nx.Graph, Tuple[torch.Tensor, torch.Tensor]],
        q: int,
        X_avoid: Optional[torch.Tensor] = None,
        stochastic: bool = False,
        num_restarts: int = 20,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        patience=50,
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
    dim = 1  # the input is in terms of the index of graph, so the dimensionality is always 1
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
                            acq_val_neighbors = objective_f(
                                neighbors.unsqueeze(1))
                            n_queries += 1
                            if acq_val_neighbors[0] <= curr_f:
                                patience -= 1
                            else:
                                curr_x, curr_f = neighbors, acq_val_neighbors[0]
                                break
                            neighbors = generate_neighbors(
                                int(x), context_graph, X_avoid)
                else:
                    with torch.no_grad():
                        acq_val_neighbors = objective_f(neighbors.unsqueeze(1))
                        n_queries += neighbors.shape[0]
                    if acq_val_neighbors.max() <= curr_f:
                        break   # local minimum reached
                    best_ind = acq_val_neighbors.argmax().item()
                    curr_x, curr_f = neighbors[best_ind].unsqueeze(
                        0), acq_val_neighbors[best_ind]
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
    def __init__(
        self,
        graph: nx.Graph,
        objective_func: Callable,
        save_path: str,
    ) -> None:
        self.graph = graph
        self.n = len(graph)
        self.ground_truth = objective_func
        # random_indices = np.random.choice(
        #     n, min(n, n_samples), replace=False).tolist()
        self.colors = [float(self.ground_truth(i)) for i in range(self.n)]
        # self.colors = [float(self.ground_truth(i)) for i in range(n**2)]

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def add_candidate(self, candidates):

        for candidate in candidates:
            self.colors[int(candidate[0])] = -3

    def make_plot(self, n_iters: int):

        cmap = plt.cm.Blues
        vmin = min(self.colors)
        vmax = max(self.colors)
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_color=self.colors, node_size=50,
                cmap=cmap, vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        # plt.colorbar(sm)
        plt.savefig(os.path.join(self.save_path,
                    f"{str(n_iters).zfill(4)}.png"))

    def make_animation(self):
        return 0


def gen_k_fold_cv_folds(
    train_X: torch.Tensor, train_Y: torch.Tensor, train_Yvar: Optional[torch.Tensor] = None, fold: int = -1
):
    """
    A generalization of the LOO-CV function in botorch by supporting k-fold CV.
    """
    # fold == -1 for LOO-CV
    if fold == -1 or fold == train_X.shape[-2]:
        return gen_loo_cv_folds(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)

    masks = torch.zeros(
        fold, train_X.shape[-2], dtype=torch.uint8, device=train_X.device)
    splitted_indices = np.array_split(np.arange(train_X.shape[-2]), fold)
    for i in range(fold):
        masks[i, splitted_indices[i].tolist()] = 1
    print(masks)
    train_X_cv = torch.cat(
        [train_X[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_X_cv = torch.cat([train_X[..., m, :].unsqueeze(dim=-3)
                           for m in masks], dim=-3)
    train_Y_cv = torch.cat(
        [train_Y[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_Y_cv = torch.cat([train_Y[..., m, :].unsqueeze(dim=-3)
                           for m in masks], dim=-3)
    if train_Yvar is None:
        train_Yvar_cv = None
        test_Yvar_cv = None
    else:
        train_Yvar_cv = torch.cat(
            [train_Yvar[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
        test_Yvar_cv = torch.cat(
            [train_Yvar[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
    return CVFolds(
        train_X=train_X_cv,
        test_X=test_X_cv,
        train_Y=train_Y_cv,
        test_Y=test_Y_cv,
        train_Yvar=train_Yvar_cv,
        test_Yvar=test_Yvar_cv,
    )
