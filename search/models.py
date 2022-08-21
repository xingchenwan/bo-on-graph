from torch import Tensor
import torch
from typing import Tuple, Union, Optional
import networkx as nx
from botorch.models import FixedNoiseGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from .kernels import DiffusionGraphKernel
from .utils import eigendecompose_laplacian
from botorch.acquisition import (ExpectedImprovement,
                                 NoisyExpectedImprovement,
                                 qExpectedImprovement,
                                 qNoisyExpectedImprovement,
                                 UpperConfidenceBound,
                                 qUpperConfidenceBound)
from botorch.sampling.samplers import SobolQMCNormalSampler
from math import log


def initialize_model(
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        context_graph: Union[nx.Graph, Tuple[Tensor, Tensor]],
        use_fixed_noise: bool = True,
        fit_model: bool = False,
        ard: bool = True,
        **optim_kwargs
):
    laplacian_eigenvals, laplacian_eigenvecs = eigendecompose_laplacian(context_graph)
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_Y, 1e-7) * train_Y.std(dim=0).pow(2)
    model_kwargs = []
    base_model_class = FixedNoiseGP if use_fixed_noise else SingleTaskGP
    for i in range(train_Y.shape[-1]):
        model_kwargs.append(
            {
                "train_X": train_X,
                "train_Y": train_Y[..., i: i + 1],
                "outcome_transform": Standardize(m=1),
                "covar_module": DiffusionGraphKernel(
                    eigenvalues=laplacian_eigenvals,
                    eigenbasis=laplacian_eigenvecs,
                    lengthscale_constraint=GreaterThan(1e-5),
                    ard=ard,
                ),
            }
        )
        if use_fixed_noise:
            model_kwargs[i]["train_Yvar"] = train_Yvar[..., i: i + 1]
        else:
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3)
            )
    models = [base_model_class(**model_kwargs[i]) for i in range(train_Y.shape[-1])]
    if len(models) > 1:
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        model = models[0]
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if fit_model:
        fit_gpytorch_torch(mll, **optim_kwargs)
    return model, mll


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
        if ucb_beta is not None:
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
        method: str = "enumerate"
):
    assert method in ["enumerate", "local_search"]
