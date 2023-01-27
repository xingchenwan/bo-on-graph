from multiprocessing.sharedctypes import Value
import networkx as nx
from typing import Optional, Any, Union, Tuple, Callable, Dict
import random
import numpy as np
import torch
from problems.base_problem import Problem
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from math import sqrt
from problems.hasse import generate_hasse, generate_jaccard
import future.utils

def get_synthetic_problem(
        label: str,
        seed: int = 0,
        n: int = 5000,
        problem_kwargs: Optional[Dict[str, Any]] = None,
) -> "SyntheticProblem":
    problem_kwargs = problem_kwargs or {}
    n = problem_kwargs.get("n", 5000)
    random_graph_type = problem_kwargs.get("random_graph_type", "ba")
    assert random_graph_type in ["ba", "ws", "grid", "seed", "sbm", "set", "jaccard", "real"]
    if random_graph_type == "ba":
        m = problem_kwargs.get("m", 1)
        g = nx.generators.random_graphs.barabasi_albert_graph(
            seed=seed, n=n, m=m)
    elif random_graph_type == "ws":
        k = problem_kwargs.get("k", 3)
        p = problem_kwargs.get("p", 0.2)
        g = nx.generators.random_graphs.watts_strogatz_graph(
            n=n, k=k, p=p, seed=seed
        )
    elif random_graph_type == "grid":
        n, m = int(sqrt(n)), int(sqrt(n))
        g = nx.generators.grid_2d_graph(n, m)
        mapping = {}
        for i in range(n):
            for j in range(m):
                mapping[(i, j)] = i * m + j
        g = nx.relabel_nodes(g, mapping)
    elif random_graph_type == "sbm":
        sizes = problem_kwargs.get("sizes")
        p = problem_kwargs.get("p")
        g = nx.generators.stochastic_block_model(sizes, p)
    elif random_graph_type == "set":
        n_individuals = problem_kwargs.get("n_individuals")
        g, forward_dict, backward_dict = generate_hasse(range(n_individuals))
    elif random_graph_type == "jaccard":
        n_individuals = problem_kwargs.get("n_individuals")
        g, forward_dict, backward_dict = generate_jaccard(range(n_individuals))
    elif random_graph_type == "real":
        g = nx.from_edgelist(np.load("./com_edge_list.npy"))
        dict_relabel = {}
        for index, node in enumerate(sorted(list(g.nodes))):
            dict_relabel[node] = index
        g = nx.relabel_nodes(g, dict_relabel)
    else:
        raise ValueError(
            f"Unknown random_graph_type = {random_graph_type}")

    if label in ["centrality", "small_ba_betweenness"]:  # alias for backward compatibilitys
        feature_name = problem_kwargs.get(
            "feature_name", "eigenvector_centrality")
        ground_truth = compute_synthetic_node_features(
            g, feature_name=feature_name,)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)
    elif label == "diffusion":

        model = ep.SIRModel(g, seed=seed)
        config = mc.Configuration()
        beta = problem_kwargs.get("beta", 1.)
        gamma = problem_kwargs.get("gamma", 0.2)
        fraction_infected = problem_kwargs.get("fraction_infected", 0.0003)
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)
        config.add_model_parameter("fraction_infected", fraction_infected)

        model.set_initial_status(config)

        ground_truth = compute_synthetic_node_features(
            g, feature_name="diffusion", model=model, **problem_kwargs)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)
    
    elif label == "diffusion_real":
        model = ep.SIRModel(g, seed=seed)
        config = mc.Configuration()
        beta = problem_kwargs.get("beta", 1.)
        gamma = problem_kwargs.get("gamma", 0.2)
        fraction_infected = problem_kwargs.get("fraction_infected", 0.0003)
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)

        config.add_model_parameter("fraction_infected", fraction_infected)
        model.set_initial_status(config)

        ground_truth = compute_synthetic_node_features(
            g, feature_name="diffusion", model=model, **problem_kwargs)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)

    elif label == "test_function":
        ground_truth = compute_synthetic_node_features(
            g, feature_name="test_function", **problem_kwargs)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)
    
    elif label == "team_opt":
        ground_truth = compute_synthetic_node_features(
            g, feature_name="team_opt", forward_dict=forward_dict, backward_dict=backward_dict, **problem_kwargs)

        def obj_func(idx): return ground_truth[idx]
        return SyntheticProblem(g, obj_func, problem_size=len(g.nodes), **problem_kwargs)

    else:
        # todo
        raise NotImplementedError(f"Problem {label} is not implemented")


class SyntheticProblem(Problem):

    is_moo = False

    def __init__(self,
                 context_graph,
                 obj_func: Callable,
                 problem_size: Optional[int] = None,
                 noise_std: Optional[float] = None,
                 negate: bool = False,
                 log: bool = False,
                 **kwargs,
                 ):
        super().__init__(context_graph, noise_std=noise_std, negate=negate, log=log)
        self.obj_func = obj_func
        self.problem_size = problem_size or len(context_graph)

    @torch.no_grad()
    def evaluate_true(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().long()
        return self.obj_func(X)

    @property
    def ground_truth(self):
        all_possible_vals = torch.arange(self.problem_size).float()
        best = self(all_possible_vals).max()
        return best


def compute_synthetic_node_features(
        input_graph: nx.Graph,
        feature_name: str = "betweenness",
        model: ep.SIRModel = None,
        forward_dict: dict = None,
        **kwargs
):
    nnodes = len(input_graph)
    if feature_name == "betweenness":
        feature = nx.betweenness_centrality(input_graph, **kwargs)
    elif feature_name == "eigenvector_centrality":
        feature = nx.eigenvector_centrality(
            input_graph, max_iter=1000, **kwargs)
    elif feature_name == "closeness":
        feature = nx.closeness_centrality(input_graph)
    elif feature_name == "diffusion":
        
        # Keep track of nodes that have been infected
        set_infected = set([node for node, nstatus in future.utils.iteritems(model.status) if nstatus == model.available_statuses['Infected']])
        set_susceptible = [node for node, nstatus in future.utils.iteritems(model.status) if nstatus == model.available_statuses['Susceptible']]

        # Get parameters
        epsilon = kwargs.get("epsilon", 0.) # Spontaneous infection
        iteration_max_diffusion = kwargs.get("iteration_diffusion", 100)
        
        
        iteration = model.iteration()
        feature = iteration['status'] # Get initial starting points for infected nodes
        
        while (iteration['node_count'][1] != 0) and (iteration['iteration'] < iteration_max_diffusion):
            iteration = model.iteration()
            current_status = iteration['status']
            
            # Introduce spontaneous infections
            n_susceptible = len(set_susceptible)
            n_drawn = int(np.random.binomial(n=n_susceptible, p=epsilon, size=1))
            list_spontaneous_infection = random.sample(set_susceptible, n_drawn)
            for spontaneous_infection in list_spontaneous_infection:
                model.status[spontaneous_infection] = 1
            
            # Add them to new infections and value function
            for key in list_spontaneous_infection:
                if key not in set_infected:
                    set_infected.add(key)
                    feature[key] = iteration['iteration'] + 1
            for key, value in current_status.items():
                if value == 1 and key not in set_infected:
                    set_infected.add(key)
                    feature[key] = iteration['iteration'] + 1

        for key, value in feature.items():
            if value != 0:
                feature[key] = (1 - (feature[key] - 1) /
                                (iteration['iteration'] + 1))**2
    elif feature_name == "test_function":
        n = kwargs.get("n", 5000)
        noise = kwargs.get("noise", 0.)
        n, m = int(sqrt(n)), int(sqrt(n))
        test_function = kwargs.get("test_function", "rosenbrock")
        feature = dict.fromkeys(range(nnodes), 0)
        if test_function == "rosenbrock":
            def test_fun(x, y): return -100 * (2*y - 4*x**2)**2 - (1 - 2*x)**2
        elif test_function == "sphere":
            def test_fun(x, y): return -y**2 - x**2
        elif test_function == "ackley":
            def test_fun(x, y):
                return (-1)*(-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20)
        else:
            raise NotImplementedError
                
        for i in range(n):
            for j in range(m):
                feature[i * m +
                        j] = (test_fun(1 - (2 / n) * i, -1 + (2 / m) * j)) + np.random.normal(loc = 0., scale=noise)
    elif feature_name == "team_opt":
        
        ## Compute netropy defined before
        n_individuals = kwargs.get("n_individuals", 10)
        n_skills = kwargs.get("n_skills", 3)
        alpha = kwargs.get("alpha", 1)
        skills = np.random.dirichlet([alpha for _ in range(n_skills)], n_individuals)
        feature = dict.fromkeys(range(nnodes), 0)
        feature[0] = -1
        for i in range(1, nnodes):
            subset_skills = skills[np.array(forward_dict[i])]
            subset_skills_avg = subset_skills.mean(axis = 0)
            entropy_skills = -subset_skills * np.log(subset_skills)
            obj_value = np.sum(-subset_skills_avg*np.log(subset_skills_avg)) - np.mean(np.sum(entropy_skills, axis = 1))
            feature[i] = obj_value
    else:
        f = getattr(nx, feature_name, None)
        if f:
            feature = f(input_graph, **kwargs)
        else:
            raise ValueError(f"Unknown feature name {feature_name}")
    # the return type is a dict: {node: feature}. Convert them to a tensor
    ret = np.zeros(nnodes)
    for node, val in feature.items():
        ret[node] = val
    ret = torch.from_numpy(ret).float()
    return ret
