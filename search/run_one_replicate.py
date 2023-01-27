from email.mime import base
from time import time

import networkx as nx
import torch
from typing import Optional, Dict, Any
import numpy as np
from problems import get_synthetic_problem
import random
from search.models import initialize_model, get_acqf, optimize_acqf
from search.utils import (
    prune_baseline,
    generate_neighbors,
    Plot_animation,
    get_context_graph,
    eigendecompose_laplacian,
)
from search.trust_region import (
    update_state,
    restart,
)
import os
from botorch.utils.transforms import standardize
import matplotlib.pyplot as plt

supported_labels = [
    "random",
    "local_search",
    "ei",
    "ei_ego_network_1",
    "dfs",
    "bfs",
    "ei_ego_network_2",
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
        max_radius: int = 10,
        context_graph_nnode_init: int = 100,
        acqf_kwargs: Optional[dict] = None,
        acqf_optim_kwargs: Optional[dict] = None,
        model_optim_kwargs: Optional[Dict[str, Any]] = None,
        trust_region_kwargs: Optional[Dict[str, Any]] = None,
        problem_kwargs: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
        save_frequency: int = 1,
        animation: bool = False,
        animation_interval: int = 20,
):
    """
    Run one replication of the a supported algorithm
    Args:
        label: str. Defines the method string.
        seed: int. The random seed
        problem_name: str. The string that identifies the problem we'd like to run on.
        save_path: str. The saving directory of the experiment
        iterations: int. The max number of iterations (in terms of number of batches) *beyond the  
            initial random initializations*.
        batch_size: int. batch size of the algorithm
        n_initial_point: int. The number of configurations at initialization to be randomly 
            initialized.
        acq_optimizer: str (applicable for BO only). The type of acquisition optimizer. options:
            "enumerate", "local_search".
            Enumeration is currently highly recommended (local search is not tested).
        max_radius: int (applicable for BO only). The maximum radius from the centre node, when
            constructing the local context graph.
        context_graph_nnode_init: int. The initial number of nodes of the context graph. When None,
            the local context graph can be the entire graph.
        acqf_kwargs: Any keyword arguments to be passed to the acquisition function constructor.
        acqf_optim_kwargs: Any keyword arguments to be passed to the acqf optimizer.
        model_kwargs: ... to be passed to the model constrcutor.
        problem_kwargs: Any arguments to be passed to the base problem constructor. See
            "./problems" to see the possible arguments for each problem.
        trust_region_kwargs: keyword arguments to be passed to the trust region constructor.
            will be ignored when the search method does not use trust regions.
            Options:
                n_nodes: the initial number of nodes in the context graph.
                n_nodes_min: the min number of nodes in the context graph. When trust region size
                    drops below this, a restart is triggered.
                trust_region_multiplier: the factor to be multiplied (or divided) on the trust
                    region size when we expand/shrink the trust region.
                succ_tol: number of successive successes to expand the trust region
                fail_tol: number of successive failures to shrink the trust region
        dtype, device: defines the device and dtype.
        save_frequency: the frequency to save intermediate results to disc
        animation: whether to generate intermediate plots during optimzation
        animation_interval: frequency of animation. Ignored when `animation` is False.
    """
    trust_region_kwargs = dict(trust_region_kwargs) or {}
    problem_kwargs = dict(problem_kwargs) or {}

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_path = os.path.join(save_path, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tkwargs = {"dtype": dtype, "device": device}
    acqf_optim_kwargs = acqf_optim_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    model_optim_kwargs = model_optim_kwargs or {}

    base_function = get_synthetic_problem(
        problem_name, seed=seed, problem_kwargs=problem_kwargs)

    context_graph_nnode_init = min(context_graph_nnode_init, len(
        base_function.context_graph)) \
        if context_graph_nnode_init \
        else len(base_function.context_graph)
    # Plot animation
    if animation:
        plot_animation = Plot_animation(
            base_function.context_graph,
            base_function.obj_func,
            save_path=os.path.join(save_path, "animations")
        )
    if seed == 9:
        all_X = torch.arange(len(base_function.context_graph)).to(torch.float)
        all_Y = base_function(all_X.reshape(-1, 1))
        eigenvals, eigenvecs = eigendecompose_laplacian(base_function.context_graph)
        plt.subplot(121)
        plt.title("Function signal")
        plt.stem(torch.abs(eigenvecs.T @ all_Y))
        plt.subplot(122)
        plt.title("Eigenvalues")
        plt.stem(eigenvals.flatten())
        plt.savefig(os.path.join(save_path, "plot_signal.png"))
        plt.clf()

    # generate initial data
    n_initial_points = n_initial_points or 20
    use_trust_region = "ego_network" in label
    candidates, trust_region_state = restart(
        base_graph=base_function.context_graph,
        n_init=n_initial_points,
        seed=seed,
        batch_size=batch_size,
        init_context_graph_size=context_graph_nnode_init,
        use_trust_region=use_trust_region,
        options=trust_region_kwargs,
    )
    n_restart = 0
    X = candidates.reshape(-1, 1).to(**tkwargs)
    Y = base_function(X).to(**tkwargs)
    X_ = X.clone()
    Y_ = Y.clone()

    if animation:
        plot_animation.add_candidate(X)
        # plot_animation.make_plot(X.shape[0])
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
        # context_graph = nx.ego_graph(
        #     base_function.context_graph, best_loc.item(), radius=ego_radius)
        context_graph = get_context_graph(
            base_function.context_graph,
            best_loc.item(),
            nnodes=context_graph_nnode_init,
        )
        X_, Y_ = prune_baseline(X_, Y_, torch.tensor(
            list(context_graph.nodes),).to(X_))
        # index lookup:local index: global index
        inverse_map_dict = dict(zip(
            list(range(context_graph.number_of_nodes())),
            list(context_graph.nodes)
        ))
        # global index -> local index
        map_dict = {v: k for k, v in inverse_map_dict.items()}
        # functions to create the indices in terms of the global graph and the local graph

        def index_remapper(x): return torch.tensor(
            [map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
        def inverse_index_remapper(x): return torch.tensor(
            [inverse_map_dict[int(i)] for i in x]).to(x).reshape(-1, x.shape[-1])
    elif label == "dfs" or label == "bfs":
        visited = set(list(X_.numpy().flatten()))
        list_stacks = []
        for i in range(batch_size):
            neighbors_current = generate_neighbors(
                    int(X_[-i]),
                    base_function.context_graph,
                    X_avoid=X
                )
            list_stacks.append(list(neighbors_current.numpy().flatten()))
    else:
        context_graph = base_function.context_graph
        # no conversion required when we have a global model
        def index_remapper(x): return x
        def inverse_index_remapper(x): return x

    # when the context graph does not change from iter to iter, we do not have to recompute the
    #   eigenbasis of the resulting Laplacian
    cached_eigenbasis = None
    use_cached_eigenbasis = True

    for i in range(existing_iterations, iterations):
        if len(X) > iterations * batch_size:
            break
        if (i+1) % 10 == 0: 
            print(
                f"Starting label {label}, seed {seed}, iteration {i+1}, "
                f"time: {time() - start_time}, "
                f"Last obj: {obj[-batch_size:]}"
                f"current best obj: {best_obj}."
            )
        if label == "random":
            candidates = torch.from_numpy(
                np.random.RandomState(seed + i).choice(
                    base_function.problem_size,
                    size=batch_size
                )
            ).reshape(-1, 1).to(**tkwargs)
        elif "local_search" in label:
            neighbors_of_best = generate_neighbors(
                best_loc,
                base_function.context_graph,
                X_avoid=X
            )
            # when we cannot find a valid point for the local search, we have reached a local minimum.
            # randomly spawn a new starting point
            if not len(neighbors_of_best):
                candidates = None
                while candidates is None or candidates.shape[0] == 0:
                    print(f"Restart triggered at iteration {len(X)}")
                    n_restart += 1
                    candidates, trust_region_state = restart(
                        base_graph=base_function.context_graph,
                        n_init=n_initial_points,
                        seed=seed + n_restart,
                        batch_size=batch_size,
                        use_trust_region=False,
                        X_avoid=X,
                        options=trust_region_kwargs,
                    )
                    candidates = candidates.reshape(-1, 1).to(X)
                    X_ = torch.zeros(0, X_.shape[1]).to(X_)
                    Y_ = torch.zeros(0, 1).to(Y_)
            else:
                candidate_idx = np.unique(np.random.RandomState(
                    i + seed).choice(len(neighbors_of_best), batch_size, )).tolist()
                candidates = neighbors_of_best[candidate_idx]
        elif label == "dfs" or label == "bfs":
            flag = 1
            for stack in list_stacks:
                flag *= len(stack)
            if flag:
                candidates = []
                for i, stack in enumerate(list_stacks):
                    element = stack.pop()
                    if element not in visited:
                        neighbors_element = generate_neighbors(
                                                                element,
                                                                base_function.context_graph,
                                                                X_avoid=X
                                                            )
                        if label == "dfs":
                            stack = stack + list(neighbors_element.numpy().flatten())
                        elif label == "dfs":
                            stack = list(neighbors_element.numpy().flatten()) + stack
                        list_stacks[i] = stack
                    candidates.append(element)
                candidates = torch.tensor(candidates).reshape(-1,1)
            else: #Restard if stuck
                candidates = None
                while candidates is None or candidates.shape[0] == 0:
                    print(f"Restart triggered at iteration {len(X)}")
                    n_restart += 1
                    candidates, trust_region_state = restart(
                        base_graph=base_function.context_graph,
                        n_init=n_initial_points,
                        seed=seed + n_restart,
                        batch_size=batch_size,
                        use_trust_region=False,
                        X_avoid=X,
                        options=trust_region_kwargs,
                    )
                    candidates = candidates.reshape(-1, 1).to(X)
                    X_ = torch.zeros(0, X_.shape[1]).to(X_)
                    Y_ = torch.zeros(0, 1).to(Y_)

            new_y = base_function(candidates)
            X = torch.cat([X, candidates], dim=0)
            Y = torch.cat([Y, new_y], dim=0)
            X_ = torch.cat([X_, candidates], dim=0)
            Y_ = torch.cat([Y_, new_y], dim=0)
            
        else:
            # when a restart is triggered
            if use_trust_region and trust_region_state.restart_triggered:
                print(f"Restart triggered at iteration {len(X)}")
                n_restart += 1
                candidates, trust_region_state = restart(
                    base_graph=base_function.context_graph,
                    n_init=n_initial_points,
                    seed=seed + n_restart,
                    batch_size=batch_size,
                    use_trust_region=use_trust_region,
                    init_context_graph_size=context_graph_nnode_init,
                    X_avoid=X,
                    options=trust_region_kwargs,
                )
                context_graph = None    # reset the context graph to be re-initialized later
                candidates = candidates.reshape(-1, 1).to(X)
                X_ = torch.zeros(0, X_.shape[1]).to(X_)
                Y_ = torch.zeros(0, 1).to(Y_)
            else:
                # create remappers to convert raw X into indices in terms of the new context graph
                X_mapped = index_remapper(X_).to(**tkwargs)
                if label == "ei_ego_network_1":
                    model, mll, cached_eigenbasis = initialize_model(
                        train_X=X_mapped,
                        train_Y=Y_,
                        context_graph=context_graph,
                        covar_type="polynomial",
                        ard=True,
                        use_fixed_noise=True,
                        use_saas_map=False,
                        fit_model=True,
                        cached_eigenbasis=cached_eigenbasis,
                        use_cached_eigenbasis=use_cached_eigenbasis,
                        optim_kwargs=model_optim_kwargs,
                    )
                elif label == "ei_ego_network_2":
                    model, mll, cached_eigenbasis = initialize_model(
                        train_X=X_mapped,
                        train_Y=Y_,
                        context_graph=context_graph,
                        covar_type="diffusion",
                        covar_kwargs = {"order": 3,}, ## No order means context graph size
                        ard=True,
                        use_fixed_noise=True,
                        use_saas_map=False,
                        fit_model=True,
                        cached_eigenbasis=cached_eigenbasis,
                        use_cached_eigenbasis=use_cached_eigenbasis,
                        optim_kwargs=model_optim_kwargs,
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
                    raise NotImplementedError()  # todo: multiobjective problems

                # generate candidates by optimizing the acqf function
                raw_candidates = optimize_acqf(
                    acq_func,
                    context_graph=context_graph,
                    method=acqf_optimizer,
                    batch_size=batch_size,
                    X_avoid=X_mapped,
                    **acqf_optim_kwargs,
                )
                if raw_candidates is None:
                    if use_trust_region:
                        trust_region_state.restart_triggered = True
                        continue
                    else:
                        break

                # the candidates are in terms of the local graph -- map them back to the global graph
                candidates = inverse_index_remapper(
                    raw_candidates).to(X).reshape(-1, 1)

        if animation and animation_interval > 0 and (X.shape[0] + 1) % animation_interval == 0:
            plot_animation.add_candidate(candidates, )
            plot_animation.make_plot(n_iters=X.shape[0] + candidates.shape[0])

        # evaluate the problem
        new_y = base_function(candidates)

        # update the trust region state, if applicable
        if use_trust_region:
            trust_region_state = update_state(
                state=trust_region_state,
                Y_next=new_y
            )

        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        X_ = torch.cat([X_, candidates], dim=0)
        Y_ = torch.cat([Y_, new_y], dim=0)

        wall_time[i] = time() - start_time

        if is_moo:
            raise NotImplemented()    # todo
        else:
            obj = Y_
            new_best_obj = obj.max().view(-1)[0].cpu()
            # when best point changes we need to recompute ego net
            if ("ego_network" in label or "local_search" in label) \
                    and (new_best_obj != best_obj or context_graph is None):
                best_idx = obj.argmax().cpu()
                best_loc = X_[best_idx]
                if "ego_network" in label:
                    context_graph = get_context_graph(
                        base_function.context_graph,
                        best_loc.item(),
                        nnodes=context_graph_nnode_init,
                    )
                    X_, Y_ = prune_baseline(X, Y, torch.tensor(
                        list(context_graph.nodes)).to(X))
                    # the context graph changed -- need to re-compute the eigenbasis for the next BO iteration.
                    inverse_map_dict = dict(zip(
                        list(range(context_graph.number_of_nodes())),
                        list(context_graph.nodes)
                    ))
                    map_dict = {v: k for k, v in inverse_map_dict.items()}
                    use_cached_eigenbasis = False
            else:
                use_cached_eigenbasis = True
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
        #print(f"Current candidate {candidates}")

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
