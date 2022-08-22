from torch import Tensor
import torch
from typing import Tuple, Union, Optional, Dict, Any
import networkx as nx
from botorch.models import FixedNoiseGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
# from botorch.optim.fit import fit_gpytorch_torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from .kernels import DiffusionGraphKernel
from .utils import eigendecompose_laplacian, local_search, fit_gpytorch_model
from botorch.acquisition import (ExpectedImprovement,
                                 NoisyExpectedImprovement,
                                 qExpectedImprovement,
                                 qNoisyExpectedImprovement,
                                 UpperConfidenceBound,
                                 qUpperConfidenceBound)
from botorch.sampling.samplers import SobolQMCNormalSampler
from math import log
import botorch

def initialize_model(
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        context_graph: nx.Graph,
        use_fixed_noise: bool = True,
        fit_model: bool = False,
        ard: bool = False,
        cached_eigenbasis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cached_eigenbasis: bool = True,
        **optim_kwargs
):
    if (not use_cached_eigenbasis) or cached_eigenbasis is None:
        laplacian_eigenvals, laplacian_eigenvecs = eigendecompose_laplacian(context_graph)
    else:
        laplacian_eigenvals, laplacian_eigenvecs = use_cached_eigenbasis
        cached_eigenbasis = (laplacian_eigenvals, laplacian_eigenvecs)
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_Y, 1e-7) * train_Y.std(dim=0).pow(2)
    model_kwargs = []
    base_model_class = FixedNoiseGP if (use_fixed_noise and not torch.isnan(train_Yvar).any()) else SingleTaskGP

    for i in range(train_Y.shape[-1]):
        model_kwargs.append(
            {
                "train_X": train_X,
                "train_Y": train_Y[..., i: i + 1],
                "outcome_transform": Standardize(m=1),
                "covar_module": gpytorch.kernels.ScaleKernel(
                    DiffusionGraphKernel(
                    eigenvalues=laplacian_eigenvals,
                    eigenbasis=laplacian_eigenvecs,
                    lengthscale_constraint=GreaterThan(1e-5),
                    ard_num_dims=len(context_graph) if ard else None,
                    )
                )
            }
        )
        if use_fixed_noise and not torch.isnan(train_Yvar).any():
            model_kwargs[i]["train_Yvar"] = train_Yvar[..., i: i + 1]
        else:
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3)
            )
    models = [base_model_class(**model_kwargs[i]) for i in range(train_Y.shape[-1])]
    if len(models) > 1:
        model = ModelListGP(*models).to(device=train_X.device)
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(device=train_X.device)
    else:
        model = models[0].to(device=train_X.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device=train_X.device)
    if fit_model:
        fit_gpytorch_model(mll, model, train_X, train_Y, **optim_kwargs)
    return model, mll, cached_eigenbasis


def get_acqf(
        model,
        X_baseline: Tensor,
        train_Y: Tensor,
        batch_size: int = 1,
        acq_type: str = "ei",
        mc_samples: int = 1024,
        ucb_beta: Optional[float] = None,
        iteration: Optional[int] = None,
        ref_point: Optional[Tensor] = None
):
    if acq_type == "ucb":
        if ucb_beta:
            beta = ucb_beta
        else:
            assert iteration is not None
            beta = 0.2 * X_baseline.shape[-1] * log(2 * iteration)

    if acq_type in ["ei", "ucb"]:
        assert train_Y.shape[1] == 1
        if batch_size == 1:
            if acq_type == "ei":
                acq_func = ExpectedImprovement(model, best_f=train_Y.max(), )
            elif acq_type == 'nei':
                acq_func = NoisyExpectedImprovement(model, X_observed=X_baseline)
            elif acq_type == "ucb":

                acq_func = UpperConfidenceBound(model, beta=beta, )
        else:
            if acq_type == "ei":
                acq_func = qExpectedImprovement(model=model,
                                                best_f=train_Y.max(),
                                                sampler=SobolQMCNormalSampler(mc_samples)
                                                )
            elif acq_type == "nei":
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X_baseline,
                    sampler=SobolQMCNormalSampler(mc_samples),
                    prune_baseline=True,
                )
            elif acq_type == "ucb":
                acq_func = qUpperConfidenceBound(model=model,
                                                 beta=beta,
                                                 sampler=SobolQMCNormalSampler(mc_samples)
                                                 )
    elif acq_type == "ehvi":
        assert ref_point is not None
        raise NotImplementedError()     # todo: support vector-valued output function.
    return acq_func


def optimize_acqf(
        acqf: botorch.acquisition.AcquisitionFunction,
        context_graph: Union[nx.Graph, Tuple[Tensor, Tensor]],
        method: str = "enumerate",
        batch_size: int = 1,
        options: Optional[Dict[str, Any]] = None,
):
    assert method in ["enumerate", "local_search"]
    nodes_to_eval = []
    if method == "enumerate":
        # enumerate the acquisition function of all nodes in the context graph
        if isinstance(context_graph, nx.Graph):
            nnodes = len(context_graph.nodes)
        else:
            nnodes = context_graph[0].shape[0]
        all_possible_nodes = torch.arange(nnodes).reshape(-1, 1)
        for q in range(batch_size):
            acqf_vals = acqf(all_possible_nodes.unsqueeze(1))
            best_node = torch.argmax(acqf_vals).item()
            nodes_to_eval.append(best_node)
            if batch_size > 1:
                acqf.set_X_pending(best_node)
    elif method == "local_search":
        default_options = {
            "num_restarts": 5,
            "device": "cpu",
        }
        default_options.update(options or {})
        nodes_to_eval, _, _ = local_search(
            acqf,
            context_graph,
            q=batch_size,
            **default_options,
        ).tolist()
    # nodes_to_eval = torch.cat(nodes_to_eval)
    return torch.tensor(nodes_to_eval)

