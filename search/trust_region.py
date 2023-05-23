from dataclasses import dataclass
import math
from typing import Any, Optional, Dict
import torch
import networkx as nx
import numpy as np
from search.utils import filter_invalid


@dataclass
class TrustRegionState:
    dim: int = 1,
    batch_size: int = 1,
    n_nodes: int = 50
    n_nodes_min: int = 5
    n_nodes_max: int = 100
    failure_counter: int = 0
    fail_tol: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    succ_tol: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    trust_region_multiplier: float = 1.5

    # def __post_init__(self):
    #     self.fail_tol = 20


def update_state(state: "TrustRegionState", Y_next: torch.Tensor,):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.succ_tol:  # Expand trust region
        state.n_nodes = int(min(state.trust_region_multiplier *
                            state.n_nodes, state.n_nodes_max))
        state.success_counter = 0
    elif state.failure_counter == state.fail_tol:  # Shrink trust region

        state.restart_triggered = True
        state.n_nodes //= state.trust_region_multiplier
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.n_nodes < state.n_nodes_min:
        state.restart_triggered = True
    return state


def restart(
    base_graph: nx.Graph,
    n_init: int,
    seed: int,
    init_context_graph_size: int = None,
    batch_size: int = 1,
    use_trust_region: bool = True,
    X_avoid: Optional[torch.Tensor] = None,
    patience: int = 50,
    options: Optional[Dict[str, Any]] = None
):
    """
    Restart function. Used at either at the initialization of optimization, or when
        a trust region restart is triggered.
    """
    # this is the kwargs options to initialize a new TrustRegionState object
    default_options = {
        "n_nodes_min": 5,
        "fail_tol": 20,
        "succ_tol": 10,
        "trust_region_multiplier": 1.5,
    }
    default_options.update(options or {})
    n_init = min(n_init, len(base_graph))

    candidates = []
    while patience and len(candidates) < n_init:
        candidates = torch.from_numpy(np.random.RandomState(seed + patience).choice(
            len(base_graph), n_init, replace=False),)
        if X_avoid is not None:
            candidates = filter_invalid(candidates, X_avoid.to(candidates))
        patience -= 1
    if len(candidates) >= n_init:
        candidates = candidates[:n_init]

    # initialize a new state
    if use_trust_region:
        current_failtol = default_options["fail_tol"]
        print(f"Here we are, the judgment day :{current_failtol}")
        current_tr = default_options["trust_region_multiplier"]
        print(f"Here we are, the judgment day :{current_tr}")
        trust_region_state = TrustRegionState(
            dim=1,
            n_nodes_max=init_context_graph_size,
            batch_size=batch_size,
            **default_options
        )
        return candidates, trust_region_state
    return candidates, None
