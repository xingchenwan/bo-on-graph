from time import time

import networkx as nx
import torch
from typing import Optional, Dict, Any
import numpy as np
from problems import get_synthetic_problem, all_synthetic_problem_labels
import random
from search.models import initialize_model, get_acqf, optimize_acqf
from search.utils import (
    prune_baseline, generate_neighbors, filter_invalid
)
import os

supported_labels = [
    "random",
    "local_search",
    "ei",
    "ei_ego_network_1",
]


def run_one_replication(
        label: str,
        seed: int,
        problem_name: str,
        save_path: str,
        iterations: int = 100,
        batch_size: int = 1,
        n_initial_points: Optional[int] = None,
        acqf_optimizer: str = "enumerate",
        acqf_kwargs: Optional[dict] = None,
        acqf_optim_kwargs: Optional[dict] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
        prune_points: bool = True,
        save_frequency: int = 1,
        ego_radius: int = 2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_path = os.path.join(save_path, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tkwargs = {"dtype": dtype, "device": device}
    acqf_optim_kwargs = acqf_optim_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    model_kwargs = model_kwargs or {}

    if problem_name in all_synthetic_problem_labels:
        base_function = get_synthetic_problem(problem_name)
    else:
        raise NotImplementedError() # todo

    # generate initial data
    n_initial_points = n_initial_points or 20
    X = torch.randint(
        0, base_function.problem_size, (n_initial_points, 1)
    ).to(**tkwargs)
    Y = base_function(X).to(**tkwargs)
    X_ = X.clone()
    Y_ = Y.clone()

    is_moo = base_function.is_moo
    if is_moo:
        raise NotImplementedError()

    # Set some counters to keep track of things.
    start_time = time()
    existing_iterations = 0
    wall_time = torch.zeros(iterations, dtype=dtype)
    if is_moo:
        pass     # todo: handle MOO
    else:
        obj = Y
    best_obj = obj.max().view(-1)[0].cpu()
    best_loc = X[obj.argmax().view(-1)[0]].cpu()
    if acqf_optimizer is None:
        acqf_optimizer = "enumerate" if base_function.problem_size <= 1000 else "local_search"

    # get the context graph
    if "ego_network" in label:
        context_graph = nx.ego_graph(base_function.context_graph, best_loc.item(), radius=ego_radius)
        if prune_points:
            X_, Y_ = prune_baseline(X_, Y_, torch.tensor(list(context_graph.nodes),).to(X_))
        # index lookup:local index: global index
        inverse_map_dict = dict(zip(
            list(range(context_graph.number_of_nodes())),
            list(context_graph.nodes)
        ))
        # global index -> local index
        map_dict = {v: k for k, v in inverse_map_dict.items()}
        # functions to create the indices in terms of the global graph and the local graph
        index_remapper = lambda x: torch.tensor([map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
        inverse_index_remapper = lambda x: torch.tensor([inverse_map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
    else:
        context_graph = base_function.context_graph
        # no conversion required when we have a global model
        index_remapper = lambda x: x
        inverse_index_remapper = lambda x: x

    # when the context graph does not change from iter to iter, we do not have to recompute the
    #   eigenbasis of the resulting Laplacian
    cached_eigenbasis = None
    use_cached_eigenbasis = True

    # methods for local search only
    cached_neighbors = None
    use_cached_neighbors = True

    for i in range(existing_iterations, iterations):
        print(
            f"Starting label {label}, seed {seed}, iteration {i}, "
            f"time: {time() - start_time}, "
            f"Last obj: {obj[-batch_size:]}"
            f"current best obj: {best_obj}."
        )
        if label == "random":
            candidates = torch.from_numpy(
                np.random.RandomState(seed+i).choice(
                    base_function.problem_size,
                    size=batch_size
                )
            ).reshape(-1, 1).to(**tkwargs)
        elif "local_search" in label:
            if use_cached_neighbors and cached_neighbors is not None:
                neighbors_of_best = filter_invalid(
                    cached_neighbors,
                    X_avoid=X)
            else:
                neighbors_of_best = generate_neighbors(
                    best_loc,
                    base_function.context_graph,
                    X_avoid=X
                )
            # when we cannot find a valid point for the local search, we have reached a local minimum.
            # randomly spawn a new starting point
            if not len(neighbors_of_best):
                patience = 50
                candidates = []
                while patience > 0 and len(candidates) < batch_size:
                    candidates = torch.from_numpy(np.random.RandomState(seed+patience).choice(
                        base_function.problem_size, batch_size)).to(**tkwargs)
                    candidates = filter_invalid(candidates, X)
                    patience -= 1
                if len(candidates) >= batch_size:
                    candidates = candidates[:batch_size]
            else:
                candidate_idx = np.unique(np.random.RandomState(i+seed).choice(len(neighbors_of_best), batch_size, )).tolist()
                candidates = neighbors_of_best[candidate_idx]
        else:
            # create remappers to convert raw X into indices in terms of the new context graph
            X_mapped = index_remapper(X_).to(**tkwargs)
            model, mll, cached_eigenbasis = initialize_model(
                train_X=X_mapped,
                train_Y=Y_,
                context_graph=context_graph,
                fit_model=True,
                cached_eigenbasis=cached_eigenbasis,
                use_cached_eigenbasis=use_cached_eigenbasis,
                **model_kwargs
            )
            if not is_moo:
                acq_func = get_acqf(
                    model,
                    X_baseline=X_mapped,
                    train_Y=Y_,
                    batch_size=batch_size,
                    acq_type="ei",
                    **acqf_kwargs
                )
            else:
                raise NotImplementedError() # todo: multiobjective problems

            # generate candidates by optimizing the acqf function
            raw_candidates = optimize_acqf(
                acq_func,
                context_graph=context_graph,
                method=acqf_optimizer,
                batch_size=batch_size,
                **acqf_optim_kwargs,
            )

            # the candidates are in terms of the local graph -- map them back to the global graph
            candidates = inverse_index_remapper(raw_candidates).to(X)

        # evaluate the problem
        new_y = base_function(candidates)

        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        X_ = torch.cat([X_, candidates], dim=0)
        Y_ = torch.cat([Y_, new_y], dim=0)

        wall_time[i] = time() - start_time

        if is_moo:
            pass    # todo
        else:
            obj = Y_
            new_best_obj = obj.max().view(-1)[0].cpu()
            # when best point changes we need to recompute ego net
            if ("ego_network" in label or "local_search" in label) \
                    and new_best_obj != best_obj:
                best_loc = X[obj.argmax().cpu()]
                context_graph = nx.ego_graph(base_function.context_graph, best_loc.item(), radius=ego_radius)
                if prune_points:
                    X_, Y_ = prune_baseline(X, Y, torch.tensor(
                        list(context_graph.nodes)).to(X))
                # the context graph changed -- need to re-compute the eigenbasis for the next BO iteration.
                inverse_map_dict = dict(zip(
                    list(range(context_graph.number_of_nodes())),
                    list(context_graph.nodes)
                ))
                map_dict = {v: k for k, v in inverse_map_dict.items()}
                use_cached_eigenbasis = False
                use_cached_neighbors = False
            else:
                use_cached_eigenbasis = True
                use_cached_neighbors = True
            best_obj = new_best_obj

            # Periodically save the output.
            if save_frequency is not None and iterations % save_frequency == 0:
                if hasattr(base_function, "ground_truth"):
                    regret = base_function.ground_truth.cpu() - Y.cpu()
                else:
                    regret = None
                output_dict = {
                    "label": label,
                    "X": X.cpu(),
                    "Y": Y.cpu(),
                    "wall_time": wall_time[: i + 1],
                    "best_obj": best_obj,
                    "regret": regret
                }
                with open(os.path.join(save_path, f"{str(seed).zfill(4)}_{label}.pt"), "wb") as fp:
                    torch.save(output_dict, fp)

    # Save the final output
    if hasattr(base_function, "ground_truth"):
        regret = base_function.ground_truth.cpu() - Y.cpu()
    else:
        regret = None
    output_dict = {
        "label": label,
        "X": X.cpu(),
        "Y": Y.cpu(),
        "wall_time": wall_time,
        "best_obj": best_obj,
        "regret": regret,
    }
    with open(os.path.join(save_path, f"{str(seed).zfill(4)}_{label}.pt"), "wb") as fp:
        torch.save(output_dict, fp)


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    save_dir = "./logs/synthetic/"
    for i in range(20):
        run_one_replication(
            "ei_ego_network_1",
            seed=i,
            problem_name="small_ba_betweenness",
            save_path=save_dir,
            batch_size=1,
            n_initial_points=10,
        )