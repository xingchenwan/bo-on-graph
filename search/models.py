import numpy as np
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
from .kernels import DiffusionGraphKernel, PolynomialKernel, PolynomialKernelNew
from .utils import eigendecompose_laplacian, local_search, fit_gpytorch_model, filter_invalid, gen_k_fold_cv_folds
from botorch.acquisition import (ExpectedImprovement,
                                 NoisyExpectedImprovement,
                                 qExpectedImprovement,
                                 qNoisyExpectedImprovement,
                                 UpperConfidenceBound,
                                 qUpperConfidenceBound)
from botorch.sampling.normal import SobolQMCNormalSampler
from math import log
import botorch
from botorch.utils.transforms import standardize


def initialize_model(
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        context_graph: nx.Graph,
        covar_type: str = "polynomial",
        covar_kwargs: Optional[Dict[str, Any]] = None,
        use_fixed_noise: bool = True,
        fit_model: bool = False,
        ard: bool = False,
        cached_eigenbasis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cached_eigenbasis: bool = True,
        use_normalized_laplacian: bool = True,
        use_normalized_eigenvalues: bool = True,
        use_saas_map: bool = False,
        n_cv_fold: int = -1,
        taus: Optional[torch.Tensor] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        standardize_y: bool = True,
):
    if (not use_cached_eigenbasis) or cached_eigenbasis is None:
        laplacian_eigenvals, laplacian_eigenvecs = eigendecompose_laplacian(
            context_graph,
            normalized_laplacian=use_normalized_laplacian,
            normalized_eigenvalues=use_normalized_eigenvalues,)
    else:
        laplacian_eigenvals, laplacian_eigenvecs = cached_eigenbasis
    cached_eigenbasis = (laplacian_eigenvals, laplacian_eigenvecs)
    if standardize_y:
        train_Y = standardize(train_Y.clone())
    if use_fixed_noise:
        train_Yvar = torch.full_like(
            train_Y, 1e-7) * train_Y.std(dim=0).pow(2)
    covar_kwargs = covar_kwargs or {}
    optim_options = dict(lr=0.1, mu_0=0.1, train_iters=100)
    optim_options.update(optim_kwargs or {})
    # MAP variant of SAAS-GP. Use a grid of \taus (that determine the scale of the Half-cauchy
    #   prior governing the lengthscales, and then run cross validation to choose the best
    #   model.
    if use_saas_map:
        cv_folds = gen_k_fold_cv_folds(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar if (
                use_fixed_noise and not torch.isnan(train_Yvar).any()) else None,
            fold=n_cv_fold)

        taus = taus if taus is not None else torch.tensor(
            [10 ** i for i in range(-4, 1)])
        errs = []
        for i, tau in enumerate(taus):
            covar_kwargs.update(
                {"lengthscale_prior": gpytorch.priors.HalfCauchyPrior(tau)})
            model, mll, _ = initialize_model(
                train_X=cv_folds.train_X,
                train_Y=cv_folds.train_Y,
                context_graph=context_graph,
                covar_type=covar_type,
                covar_kwargs=covar_kwargs,
                use_fixed_noise=use_fixed_noise,
                fit_model=True,
                ard=ard,
                cached_eigenbasis=cached_eigenbasis,
                use_cached_eigenbasis=True,
                use_normalized_laplacian=use_normalized_eigenvalues,
                use_normalized_eigenvalues=use_normalized_eigenvalues,
                use_saas_map=False,
                optim_kwargs=optim_kwargs
            )
            # compute the LOO/k-fold-CV error and use it as the cost function for model selection
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(
                    cv_folds.test_X,
                )
                mean = posterior.mean
                cv_error = ((cv_folds.test_Y.squeeze() -
                            mean.squeeze()) ** 2).mean()
                errs.append(cv_error)
            # print(f"Tau = {tau}. CV error ={cv_error.item()}")
        best_idx = torch.stack(errs).argmin().item()
        # choose the best
        tau_best = taus[best_idx]
        # retrain on the full data
        covar_kwargs.update(
            {"lengthscale_prior": gpytorch.priors.HalfCauchyPrior(tau_best)})
        model, mll, _ = initialize_model(
            train_X=train_X,
            train_Y=train_Y,
            context_graph=context_graph,
            covar_type=covar_type,
            covar_kwargs=covar_kwargs,
            use_fixed_noise=use_fixed_noise,
            fit_model=True,
            ard=ard,
            cached_eigenbasis=cached_eigenbasis,
            use_cached_eigenbasis=True,
            use_normalized_laplacian=use_normalized_eigenvalues,
            use_normalized_eigenvalues=use_normalized_eigenvalues,
            use_saas_map=False,
            optim_kwargs=optim_kwargs
        )
        return model, mll, cached_eigenbasis

    else:
        model_kwargs = []
        base_model_class = FixedNoiseGP if (
            use_fixed_noise and not torch.isnan(train_Yvar).any()) else SingleTaskGP
        covar_kwargs = covar_kwargs or {}
        if covar_type in ["polynomial", "diffusion", "polynomial_new"]:
            if covar_type == "polynomial":
                base_covar_class = PolynomialKernel
            elif covar_type == "diffusion":
                base_covar_class = DiffusionGraphKernel
            else:
                base_covar_class = PolynomialKernelNew
            order = covar_kwargs.get("order", None)
            # when order is not explicitly specified,
            if covar_type == "diffusion":
                order = min(
                    order, train_X.shape[-2]) if order else len(context_graph)
            elif covar_type in ["polynomial", "polynomial_new"]::
                if order == None:
                    order = min(5, nx.radius(context_graph))
                
                #order = min(order, nx.radius(context_graph)
                #            ) if order else min(5, nx.radius(context_graph))
            if ard:
                ard_num_dims = order
            else:
                ard_num_dims = None
            covar_kwargs.update({"ard_num_dims": ard_num_dims, "order": order})
        else:
            raise NotImplementedError(
                f"covar_type {covar_type} is not implemented.")

        if "lengthscale_constraint" not in covar_kwargs.keys():
            covar_kwargs["lengthscale_constraint"] = GreaterThan(1e-5)

        if train_Y.shape[-1] > 1:
            raise NotImplementedError(f"Multi - objective search is not currently supported."
                                      "train_Y has last dimension"
                                      "of {train_Y.shape[-1]}!")
        model_kwargs.append(
            {
                "train_X": train_X,
                "train_Y": train_Y,
                # "outcome_transform": Standardize(m=1),
                "covar_module":
                gpytorch.kernels.ScaleKernel(
                    base_covar_class(
                        eigenvalues=laplacian_eigenvals,
                        eigenbasis=laplacian_eigenvecs,
                        **covar_kwargs
                    )
                )
            }
        )
        if use_fixed_noise and not torch.isnan(train_Yvar).any():
            model_kwargs[0]["train_Yvar"] = train_Yvar
        else:
            model_kwargs[0]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3)
            )

        # create model
        models = [base_model_class(**model_kwargs[i])
                  for i in range(len(model_kwargs))]
        if len(models) > 1:
            model = ModelListGP(*models).to(device=train_X.device)
            mll = SumMarginalLogLikelihood(
                model.likelihood, model).to(device=train_X.device)
        else:
            model = models[0].to(device=train_X.device)
            mll = ExactMarginalLogLikelihood(
                model.likelihood, model).to(device=train_X.device)
        if fit_model:
            if covar_type in ["polynomial", "diffusion"]:
                with gpytorch.settings.debug(False):
                    fit_gpytorch_model(mll, model, train_X,
                                       train_Y, **optim_options)
            else:
                fit_gpytorch_model_with_constraints(
                    mll, model, train_X, train_Y, **optim_options)

    return model, mll, cached_eigenbasis

def fit_gpytorch_model_with_constraints(
    mll, model, train_x, train_Y, train_iters: int = 100,
    lr: float = 0.1,
    print_interval: int = 10,
    mu_0: float = 0.01,
    return_loss: bool = False
):
    r"""Fit the GP model for the new polynomial kernel. Note that for this model,
    we do not constrain individual ``\betas` to be non-negative and the only
    constraint is that they sum to positive. That is:
    Optimize log likelihood
    s.t. \betaB > 0.
    To solve the constrained optimization problem, we use a barrier function as
    additional penalty term on the log likelihood to penalize violation of the
    constraints.
    """

    # Pre-compute the Lambda (eigenvalue and their power)
    eigen_powered = torch.cat(
        [(model.covar_module.base_kernel.eigenvalues ** i).reshape(1, -1)
         for i in range(model.covar_module.base_kernel.order)]
    )  # Shape (order, n)

    def mu_scheduler(current_step: int, total_step: int, mu_init: float):
        """Simple linear annealing (to 0) for the mu (the coeffient on the barriers).
        """
        return (1. - (current_step / total_step)) * mu_init

    with gpytorch.settings.debug(False):
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_iters)
        model.train()
        model.likelihood.train()
        for i in range(train_iters):
            # Get the constant before the barrier function
            mu = mu_scheduler(i, train_iters, mu_0)
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, model.train_targets)

            # Calculate the penalty
            constraint = [torch.sum(model.covar_module.base_kernel.beta *
                                    eigen_powered[:, j]).reshape(-1) for j in range(eigen_powered.shape[1])]
            constraint = torch.cat(constraint)
            penalty = torch.log(constraint).sum()

            if loss.ndim > 0:
                loss = loss.sum()
            if print_interval > 0 and (i + 1) % print_interval == 0:
                print(f"Iter {i+1}/{train_iters}: "
                      f"Loss={loss.item()}. Penalty={mu * penalty}"
                      f"beta={model.covar_module.base_kernel.beta}")
            loss -= mu * penalty
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
    if return_loss:
        return model, loss.item()
    return model

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
                acq_func = NoisyExpectedImprovement(
                    model, X_observed=X_baseline)
            elif acq_type == "ucb":

                acq_func = UpperConfidenceBound(model, beta=beta, )
        else:
            if acq_type == "ei":
                acq_func = qExpectedImprovement(model=model,
                                                best_f=train_Y.max(),
                                                sampler=SobolQMCNormalSampler(
                                                    mc_samples)
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
                                                 sampler=SobolQMCNormalSampler(
                                                     mc_samples)
                                                 )
    elif acq_type == "ehvi":
        assert ref_point is not None
        # todo: support vector-valued output function.
        raise NotImplementedError()
    return acq_func


def optimize_acqf(
        acqf: botorch.acquisition.AcquisitionFunction,
        context_graph: Union[nx.Graph, Tuple[Tensor, Tensor]],
        method: str = "enumerate",
        batch_size: int = 1,
        options: Optional[Dict[str, Any]] = None,
        X_avoid: Optional[torch.Tensor] = None,
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
        if X_avoid is not None:
            all_possible_nodes = filter_invalid(all_possible_nodes, X_avoid)
            if not all_possible_nodes.shape[0]:
                return None
        for q in range(batch_size):
            acqf_vals = acqf(all_possible_nodes.unsqueeze(1))
            best_node = torch.argmax(acqf_vals).item()
            nodes_to_eval.append(all_possible_nodes[best_node])
            if batch_size > 1:
                acqf.set_X_pending(all_possible_nodes[best_node])
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
