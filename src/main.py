from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ~6 months in seconds (matches C++ default 6*30 days)
DEFAULT_BIN_SIZE_SECONDS = 6 * 30 * 24 * 3600


@dataclass
class DPFHyperParams:
    """
    Hyperparameters for Dynamic Poisson Factorization.

    - K: number of latent factors
    - prior_var_dynamic / prior_var_static: variance of the Gaussian priors
      on dynamic (u_t, v_t) and static (u_bar, v_bar) log-factors.
    """

    K: int = 5
    prior_var_dynamic: float = 10.0
    prior_var_static: float = 10.0


def infer_num_users_items_from_interactions(
    train: Interactions,
    val: Interactions,
    test: Interactions,
) -> Tuple[int, int]:
    """
    Infer num_users and num_items from the three splits.
    Assumes user_ids and item_ids are already 0..N-1 and 0..M-1.
    """
    all_users = torch.cat([train.user_ids, val.user_ids, test.user_ids])
    all_items = torch.cat([train.item_ids, val.item_ids, test.item_ids])

    num_users = int(all_users.max().item()) + 1
    num_items = int(all_items.max().item()) + 1
    return num_users, num_items


class DynamicPoissonFactorization(nn.Module):
    """
    Dynamic Poisson Factorization model, variational parameters only.

    We keep:
        u[n, k, t], v[m, k, t]         : dynamic log-factors
        u_bar[n, k], v_bar[m, k]       : static log-factors

    For each, we have a Gaussian variational posterior:
        q(x) = N(mu_x, var_x)

    Here we parameterize by (mu, logvar) so that var = exp(logvar).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_times: int,
        hyper: Optional[DPFHyperParams] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if hyper is None:
            hyper = DPFHyperParams()

        self.N = num_users  # ty: ignore[unresolved-attribute]
        self.M = num_items  # ty: ignore[unresolved-attribute]
        self.T = num_times  # ty: ignore[unresolved-attribute]
        self.K = hyper.K  # ty: ignore[unresolved-attribute]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device  # ty: ignore[unresolved-attribute]

        # Store prior variances as buffers (not trainable, but move with .to(device))
        self.register_buffer(
            "prior_var_dynamic",
            torch.tensor(hyper.prior_var_dynamic, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "prior_var_static",
            torch.tensor(hyper.prior_var_static, dtype=torch.float32, device=device),
        )

        # --- Initialization helpers

        def randn_small(shape):
            """
            Small random init for means, centered near zero.
            """
            return 0.01 * torch.randn(*shape, device=device)

        # Start log-variances around log(1.0) => posterior var ~ 1 at the beginning.
        init_logvar_dyn = torch.log(torch.ones(1, device=device))
        init_logvar_static = torch.log(torch.ones(1, device=device))

        # --- Dynamic user/item factors: mu_u[n, k, t], mu_v[m, k, t]

        self.mu_u = nn.Parameter(randn_small((self.N, self.K, self.T)))
        self.mu_v = nn.Parameter(randn_small((self.M, self.K, self.T)))

        self.logvar_u = nn.Parameter(
            init_logvar_dyn.expand(self.N, self.K, self.T).clone()
        )
        self.logvar_v = nn.Parameter(
            init_logvar_dyn.expand(self.M, self.K, self.T).clone()
        )

        # --- Static user/item factors: mu_u_bar[n, k], mu_v_bar[m, k]

        self.mu_u_bar = nn.Parameter(randn_small((self.N, self.K)))
        self.mu_v_bar = nn.Parameter(randn_small((self.M, self.K)))

        self.logvar_u_bar = nn.Parameter(
            init_logvar_static.expand(self.N, self.K).clone()
        )
        self.logvar_v_bar = nn.Parameter(
            init_logvar_static.expand(self.M, self.K).clone()
        )

    def expected_user_factors(
        self, user_ids: torch.LongTensor, time_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute E[theta_{nkt}] where
            theta_{nkt} = exp(u_bar[n,k] + u[n,k,t])

        user_ids, time_ids: shape (nnz,)
        returns: tensor of shape (nnz, K)
        """
        # Gather relevant means and variances
        mu_dyn = self.mu_u[user_ids, :, time_ids]  # (nnz, K)
        mu_static = self.mu_u_bar[user_ids, :]  # (nnz, K)

        var_dyn = torch.exp(self.logvar_u[user_ids, :, time_ids])  # (nnz, K)
        var_static = torch.exp(self.logvar_u_bar[user_ids, :])  # (nnz, K)

        mu_total = mu_dyn + mu_static
        var_total = var_dyn + var_static  # independent Gaussians

        # E[exp(X)] for X ~ N(mu, var) is exp(mu + 0.5 * var)
        return torch.exp(mu_total + 0.5 * var_total)

    def expected_item_factors(
        self, item_ids: torch.LongTensor, time_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute E[beta_{mkt}] where
            beta_{mkt} = exp(v_bar[m,k] + v[m,k,t])

        item_ids, time_ids: shape (nnz,)
        returns: tensor of shape (nnz, K)
        """
        mu_dyn = self.mu_v[item_ids, :, time_ids]  # (nnz, K)
        mu_static = self.mu_v_bar[item_ids, :]  # (nnz, K)

        var_dyn = torch.exp(self.logvar_v[item_ids, :, time_ids])  # (nnz, K)
        var_static = torch.exp(self.logvar_v_bar[item_ids, :])  # (nnz, K)

        mu_total = mu_dyn + mu_static
        var_total = var_dyn + var_static

        return torch.exp(mu_total + 0.5 * var_total)

    def poisson_rate(
        self,
        user_ids: torch.LongTensor,
        item_ids: torch.LongTensor,
        time_ids: torch.LongTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute lam_{nmt} = E[ sum_k theta_{nkt} beta_{mkt} ] under q.

        user_ids, item_ids, time_ids: (nnz,)
        returns: lam shape (nnz,)
        """
        theta = self.expected_user_factors(user_ids, time_ids)  # (nnz, K)
        beta = self.expected_item_factors(item_ids, time_ids)  # (nnz, K)
        lam = (theta * beta).sum(dim=-1)  # (nnz,)
        return lam.clamp_min(eps)

    def total_expected_rate(self, max_time: Optional[int] = None) -> torch.Tensor:
        """
        Compute sum_{n,m,t} E[lambda_{nmt}] up to max_time (exclusive).
        This adds back the Poisson penalty for all zero entries.
        """
        if max_time is None:
            max_time = self.T

        # Expected exp(user log-factor) for all users/items at each time.
        mu_u_total = self.mu_u[:, :, :max_time] + self.mu_u_bar[:, :, None]  # (N,K,T)
        var_u_total = (
            torch.exp(self.logvar_u[:, :, :max_time])
            + torch.exp(self.logvar_u_bar)[:, :, None]
        )
        theta = torch.exp(mu_u_total + 0.5 * var_u_total)  # (N,K,T)

        mu_v_total = self.mu_v[:, :, :max_time] + self.mu_v_bar[:, :, None]  # (M,K,T)
        var_v_total = (
            torch.exp(self.logvar_v[:, :, :max_time])
            + torch.exp(self.logvar_v_bar)[:, :, None]
        )
        beta = torch.exp(mu_v_total + 0.5 * var_v_total)  # (M,K,T)

        # Sum over users/items, then over factors and time.
        theta_sum = theta.sum(dim=0)  # (K,T)
        beta_sum = beta.sum(dim=0)  # (K,T)
        return (theta_sum * beta_sum).sum()

    def expected_log_likelihood(self, interactions: Interactions) -> torch.Tensor:
        """
        phi-based lower bound on E_q[log p(Y | Z)] as in dPF:

          For each non-zero y_{nmt},

            E_q[log p(y_{nmt} | lam_{nmt})]
            ≳ y_{nmt} * log sum_k exp(E[log theta_{nkt}] + E[log beta_{mkt}])
               - E_q[lam_{nmt}] - log(y_{nmt}!)

          where lam_{nmt} = sum_k theta_{nkt} beta_{mkt} and expectations are
          taken under the variational Gaussians (u, u_bar, v, v_bar).

        We compute E_q[log theta] and E_q[log beta] from the means of the
        corresponding Gaussians, and E_q[lam] via `poisson_rate(...)`.
        """
        user_ids = interactions.user_ids.to(self.device).long()
        item_ids = interactions.item_ids.to(self.device).long()
        time_ids = interactions.time_ids.to(self.device).long()
        y = interactions.counts.to(self.device)

        # E[log theta_{nkt}] and E[log beta_{mkt}] (log of lognormal is just its mean)
        log_theta = self.expected_log_user_factors(
            user_ids,  # type:ignore
            time_ids,  # type:ignore
        )  # (nnz, K)
        log_beta = self.expected_log_item_factors(
            item_ids,  # type:ignore
            time_ids,  # type:ignore
        )  # (nnz, K)

        # This is log sum_k exp(E[log theta] + E[log beta]) for each (n,m,t)
        log_lam_tilde = torch.logsumexp(log_theta + log_beta, dim=-1)  # (nnz,)

        # Poisson log-likelihood over all entries:
        #   sum_{obs} y log lam - log(y!)  - sum_{all} E[lam]
        # The global -E[lam] term penalizes unobserved cells too.
        y_log_term = (y * log_lam_tilde - torch.lgamma(y + 1.0)).sum()

        # Restrict penalty to timesteps present in this split to mirror C++,
        # which only runs up to the last train timestep.
        max_time = int(interactions.time_ids.max().item()) + 1
        total_rate = self.total_expected_rate(max_time=max_time)

        log_lik = y_log_term - total_rate
        return log_lik

    def gaussian_prior_log_prob(self) -> torch.Tensor:
        """
        E_q[log p(Z)] where:
          - Dynamic u, v follow a Gaussian random walk across time.
          - Static u_bar, v_bar are IID N(0, prior_var_static).

        We assume q factorizes over all scalars, so expectations are
        simple functions of means and variances.
        """
        log2pi = math.log(2.0 * math.pi)

        sigma2_dyn = self.prior_var_dynamic  # scalar tensor
        sigma2_static = self.prior_var_static

        # --- dynamic user factors u[n,k,t]
        mu_u = self.mu_u
        var_u = torch.exp(self.logvar_u)  # same shape

        # initial time step t = 0: u_{nk0} ~ N(0, sigma2_dyn)
        term_u0 = -0.5 * (
            log2pi
            + torch.log(sigma2_dyn)  # type:ignore
            + (var_u[:, :, 0] + mu_u[:, :, 0] ** 2) / sigma2_dyn
        )  # (N, K)

        # transitions u_{nkt} | u_{nk,t-1} ~ N(u_{nk,t-1}, sigma2_dyn)
        if self.T > 1:
            delta_mu_u = mu_u[:, :, 1:] - mu_u[:, :, :-1]  # (N, K, T-1)
            var_u_t = var_u[:, :, 1:]
            var_u_prev = var_u[:, :, :-1]

            term_u_trans = -0.5 * (
                log2pi
                + torch.log(sigma2_dyn)  # type:ignore
                + (var_u_t + var_u_prev + delta_mu_u**2) / sigma2_dyn
            )  # (N, K, T-1)

            logp_u_dyn = term_u0.sum() + term_u_trans.sum()
        else:
            logp_u_dyn = term_u0.sum()

        # --- dynamic item factors v[m,k,t]
        mu_v = self.mu_v
        var_v = torch.exp(self.logvar_v)

        term_v0 = -0.5 * (
            log2pi
            + torch.log(sigma2_dyn)  # type:ignore
            + (var_v[:, :, 0] + mu_v[:, :, 0] ** 2) / sigma2_dyn
        )  # (M, K)

        if self.T > 1:
            delta_mu_v = mu_v[:, :, 1:] - mu_v[:, :, :-1]
            var_v_t = var_v[:, :, 1:]
            var_v_prev = var_v[:, :, :-1]

            term_v_trans = -0.5 * (
                log2pi
                + torch.log(sigma2_dyn)  # type:ignore
                + (var_v_t + var_v_prev + delta_mu_v**2) / sigma2_dyn
            )

            logp_v_dyn = term_v0.sum() + term_v_trans.sum()
        else:
            logp_v_dyn = term_v0.sum()

        # --- static user factors u_bar[n,k] ~ N(0, sigma2_static)
        mu_ubar = self.mu_u_bar
        var_ubar = torch.exp(self.logvar_u_bar)

        term_ubar = -0.5 * (
            log2pi + torch.log(sigma2_static) + (var_ubar + mu_ubar**2) / sigma2_static  # type:ignore
        )  # (N, K)
        logp_ubar = term_ubar.sum()

        # --- static item factors v_bar[m,k] ~ N(0, sigma2_static)
        mu_vbar = self.mu_v_bar
        var_vbar = torch.exp(self.logvar_v_bar)

        term_vbar = -0.5 * (
            log2pi + torch.log(sigma2_static) + (var_vbar + mu_vbar**2) / sigma2_static  # type:ignore
        )  # (M, K)
        logp_vbar = term_vbar.sum()

        return logp_u_dyn + logp_v_dyn + logp_ubar + logp_vbar

    def gaussian_entropy(self) -> torch.Tensor:
        """
        E_q[log q(Z)] for factorized Gaussians.
        For each scalar x ~ N(mu, sigma^2):
            E[log q(x)] = -0.5 * (log(2pi) + log sigma^2 + 1).
        """
        log2pi = math.log(2.0 * math.pi)

        eq_log_q_u = -0.5 * (log2pi + self.logvar_u + 1.0)
        eq_log_q_v = -0.5 * (log2pi + self.logvar_v + 1.0)
        eq_log_q_ubar = -0.5 * (log2pi + self.logvar_u_bar + 1.0)
        eq_log_q_vbar = -0.5 * (log2pi + self.logvar_v_bar + 1.0)

        return (
            eq_log_q_u.sum()
            + eq_log_q_v.sum()
            + eq_log_q_ubar.sum()
            + eq_log_q_vbar.sum()
        )

    def elbo(self, interactions: Interactions) -> torch.Tensor:
        """
        Compute a differentiable ELBO approximation:
          ELBO ~= E_q[log p(Y | Z)] + E_q[log p(Z)] - E_q[log q(Z)]
        """
        log_lik = self.expected_log_likelihood(interactions)
        log_p = self.gaussian_prior_log_prob()
        eq_log_q = self.gaussian_entropy()
        return log_lik + log_p - eq_log_q

    def expected_log_user_factors(
        self, user_ids: torch.LongTensor, time_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute E[log theta_{nkt}] = E[u_bar[n,k] + u[n,k,t]] under q.
        Since the variational posterior is Gaussian on these logs,
        this is just the sum of the means.
        user_ids, time_ids: shape (nnz,)
        returns: tensor of shape (nnz, K)
        """
        mu_dyn = self.mu_u[user_ids, :, time_ids]  # (nnz, K)
        mu_static = self.mu_u_bar[user_ids, :]  # (nnz, K)
        return mu_dyn + mu_static

    def expected_log_item_factors(
        self, item_ids: torch.LongTensor, time_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute E[log beta_{mkt}] = E[v_bar[m,k] + v[m,k,t]] under q.
        """
        mu_dyn = self.mu_v[item_ids, :, time_ids]  # (nnz, K)
        mu_static = self.mu_v_bar[item_ids, :]  # (nnz, K)
        return mu_dyn + mu_static


def train_dpf(
    model: DynamicPoissonFactorization,
    train_int: Interactions,
    val_int: Optional[Interactions] = None,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    optimizer_name: str = "lbfgs",
    report_every: int = 10,
    min_epochs_before_stop: int = 30,
    verbose: bool = True,
) -> Dict[str, List[Optional[float]]]:
    """
    Full-batch training loop on 'train_int' with optional validation-based early stopping
    (mirrors the C++ schedule: report every 10 iters, stop on flat/decreasing val pred LL).

    - optimizer_name: "adamw" or "lbfgs"
    - full dataset is used per step (possible to add mini-batching?).
    - if val_int is provided, predictive LL is computed every epoch and stored,
      and early stopping uses values at reporting steps (every 'report_every' epochs).
    Returns a history dict with keys:
        - "epoch"        : epoch indices
        - "elbo"         : train ELBO values
        - "val_pred_ll"  : validation predictive log-likelihood (or None if val_int is None)
    """
    model.train()
    params = list(model.parameters())
    device = model.device

    # Per-user last train time for clamping validation/test (mirrors C++ usage).
    last_train_time_cpu = last_training_timestep(train_int, model.N)
    last_train_time = torch.from_numpy(last_train_time_cpu).to(device)

    def predictive_log_likelihood(intx: Interactions) -> float:
        """
        Predictive log-likelihood over provided entries only
        (no all-zero penalty), clamping timesteps to last train
        per-user as in C++.
        """
        if intx.user_ids.numel() == 0:
            return 0.0

        user_ids = intx.user_ids.to(device).long()
        item_ids = intx.item_ids.to(device).long()
        time_ids_orig = intx.time_ids.to(device).long()
        # clamp to last observed train timestep for each user
        time_ids = torch.minimum(time_ids_orig, last_train_time[user_ids])

        y = intx.counts.to(device)
        lam = model.poisson_rate(user_ids, item_ids, time_ids)  # type:ignore
        ll = (y * torch.log(lam) - lam - torch.lgamma(y + 1.0)).sum()
        return float(ll.detach().cpu().item())

    # history containers
    history: Dict[str, List[Optional[float]]] = {
        "epoch": [],
        "elbo": [],
        "val_pred_ll": [],
    }

    prev_val_ll: Optional[float] = None
    nh = 0  # number of consecutive decreases
    stop = False

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            elbo = model.elbo(train_int)
            loss = -elbo
            loss.backward()
            optimizer.step()

            # --- compute validation predictive LL for this epoch (if any) ---
            if val_int is not None:
                val_ll = predictive_log_likelihood(val_int)
            else:
                val_ll = None

            # --- log history ---
            history["epoch"].append(float(epoch))
            history["elbo"].append(float(elbo.item()))
            history["val_pred_ll"].append(val_ll)

            # --- reporting / early stopping (same schedule as before) ---
            should_report = verbose and (
                epoch % report_every == 0 or epoch == num_epochs - 1
            )
            if should_report:
                msg = f"[AdamW] epoch={epoch:04d}  ELBO={elbo.item():.3e}"
                if val_ll is not None:
                    msg += f"  val_pred_ll={val_ll:.3e}"
                    if epoch >= min_epochs_before_stop:
                        if prev_val_ll is not None:
                            if (
                                val_ll > prev_val_ll
                                and prev_val_ll != 0
                                and abs((val_ll - prev_val_ll) / prev_val_ll) < 1e-6
                            ):
                                stop = True
                            elif val_ll < prev_val_ll:
                                nh += 1
                            else:
                                nh = 0
                            if nh > 2:
                                stop = True
                        prev_val_ll = val_ll
                print(msg)

            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        for epoch in range(num_epochs):

            def closure():
                optimizer.zero_grad()
                elbo = model.elbo(train_int)
                loss = -elbo
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            # Recompute ELBO for logging
            with torch.no_grad():
                elbo = model.elbo(train_int)

            # Validation predictive LL for this epoch (if any)
            if val_int is not None:
                val_ll = predictive_log_likelihood(val_int)
            else:
                val_ll = None

            # Log history
            history["epoch"].append(float(epoch))
            history["elbo"].append(float(elbo.item()))
            history["val_pred_ll"].append(val_ll)

            should_report = verbose and (
                epoch % report_every == 0 or epoch == num_epochs - 1
            )
            if should_report:
                msg = (
                    f"[LBFGS] epoch={epoch:04d}  loss={loss.item():.3e}  "
                    f"ELBO={elbo.item():.3e}"
                )
                if val_ll is not None:
                    msg += f"  val_pred_ll={val_ll:.3e}"
                    if epoch >= min_epochs_before_stop:
                        if prev_val_ll is not None:
                            if (
                                val_ll > prev_val_ll
                                and prev_val_ll != 0
                                and abs((val_ll - prev_val_ll) / prev_val_ll) < 1e-6
                            ):
                                stop = True
                            elif val_ll < prev_val_ll:
                                nh += 1
                            else:
                                nh = 0
                            if nh > 2:
                                stop = True
                        prev_val_ll = val_ll
                print(msg)

            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")

    return history


def build_user_time_positives(
    interactions: Interactions,
) -> dict[tuple[int, int], set[int]]:
    """
    Map (user, time_bin) -> set of positive items.
    """
    user_ids = interactions.user_ids.cpu().numpy()
    item_ids = interactions.item_ids.cpu().numpy()
    time_ids = interactions.time_ids.cpu().numpy()
    counts = interactions.counts.cpu().numpy()

    ut_pos: dict[tuple[int, int], set[int]] = defaultdict(set)
    for u, i, t, c in zip(user_ids, item_ids, time_ids, counts):
        if c > 0:
            ut_pos[(int(u), int(t))].add(int(i))
    return ut_pos


def last_training_timestep(train_int: Interactions, num_users: int) -> np.ndarray:
    """
    For each user, return the last time bin (inclusive) seen in training.
    """
    last_t = np.zeros(num_users, dtype=np.int64)
    if train_int.user_ids.numel() == 0:
        return last_t
    users = train_int.user_ids.cpu().numpy()
    times = train_int.time_ids.cpu().numpy()
    for u, t in zip(users, times):
        if t > last_t[u]:
            last_t[u] = t
    return last_t


def scores_for_user_time(
    model: DynamicPoissonFactorization,
    user_id: int,
    time_id: int,
) -> torch.Tensor:
    """
    Compute lam_{user_id, m, time_id} for all items m.

    returns: tensor of shape (M,) on CPU.
    """
    device = model.device
    M = model.M

    user_ids = torch.full((M,), user_id, dtype=torch.long, device=device)
    item_ids = torch.arange(M, dtype=torch.long, device=device)
    time_ids = torch.full((M,), time_id, dtype=torch.long, device=device)

    with torch.no_grad():
        lam = model.poisson_rate(user_ids, item_ids, time_ids)  # type:ignore

    return lam.cpu()


def paper_ranking_metrics(
    model: DynamicPoissonFactorization,
    train_int: Interactions,
    val_int: Interactions,
    test_int: Interactions,
    T: int = 50,
    test_users: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    C++-aligned ranking metrics:
      - rank items for each user at their last training timestep
      - exclude any item seen in train or validation for that user
      - only test events at or after that timestep are relevant
      - rank is truncated to top-100 (items outside contribute 0 to MRR/NDCG and
        get a rank of 101 for MAR)

    Returns averages across evaluated users:
      Recall@T  (hits within top-T divided by min(T, #relevant))
      NDCG      normalized DCG@100 with gains 2^rel - 1 and natural log
      MRR       sum of reciprocal ranks for items retrieved within top-100
      MAR       sum of ranks for relevant items (rank=101 if not in top-100)
    """
    import numpy as np

    device = model.device
    max_k = 100
    last_t = last_training_timestep(train_int, model.N)

    def gather_seen(intx: Interactions) -> dict[int, set[int]]:
        seen: dict[int, set[int]] = defaultdict(set)
        if intx.user_ids.numel() == 0:
            return seen
        for u, m in zip(intx.user_ids.cpu().numpy(), intx.item_ids.cpu().numpy()):
            seen[int(u)].add(int(m))
        return seen

    train_seen = gather_seen(train_int)
    val_seen = gather_seen(val_int)

    test_by_user: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for u, m, t, c in zip(
        test_int.user_ids.cpu().numpy(),
        test_int.item_ids.cpu().numpy(),
        test_int.time_ids.cpu().numpy(),
        test_int.counts.cpu().numpy(),
    ):
        test_by_user[int(u)].append((int(m), int(t), float(c)))

    recall_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    mar_sum = 0.0
    n_effective = 0

    ln2 = math.log(2.0)

    def _gain(x: float) -> float:
        if x <= 0:
            return 0.0
        return math.expm1(min(x * ln2, 700.0))  # clamp exponent to avoid inf

    users_to_eval: Iterable[int]
    if test_users is not None:
        users_to_eval = [int(u) for u in test_users.tolist()]
    else:
        users_to_eval = list(test_by_user.keys())

    for u in users_to_eval:
        if u >= model.N:
            continue
        t_last = int(last_t[u])

        # Keep earliest test event for each item at or after last train timestep
        rel: dict[int, float] = {}
        rel_time: dict[int, int] = {}
        for m, t, c in test_by_user.get(u, []):
            if t < t_last:
                continue
            if m not in rel_time or t < rel_time[m]:
                rel_time[m] = t
                rel[m] = c

        if not rel:
            continue

        M = model.M
        user_ids = torch.full((M,), u, dtype=torch.long, device=device)
        item_ids = torch.arange(M, dtype=torch.long, device=device)
        time_ids = torch.full(
            (M,), min(t_last, model.T - 1), dtype=torch.long, device=device
        )

        with torch.no_grad():
            scores = model.poisson_rate(user_ids, item_ids, time_ids)  # type:ignore

        # Mask seen items (train + validation)
        seen_items = train_seen.get(u, set()) | val_seen.get(u, set())
        if seen_items:
            mask = torch.full_like(scores, False, dtype=torch.bool)
            idx = torch.tensor(list(seen_items), dtype=torch.long, device=device)
            mask[idx] = True
            scores = scores.masked_fill(mask, -float("inf"))

        topk = torch.topk(scores, max_k).indices.cpu().numpy().tolist()
        ranks_map = {m: rank + 1 for rank, m in enumerate(topk)}

        pos_ranks = np.array(
            [ranks_map.get(m, np.inf) for m in rel.keys()], dtype=np.float64
        )
        num_pos = len(rel)

        # Recall@T with truncation: only top-T can be hits
        hits = float(np.count_nonzero(pos_ranks <= T))
        recall_i = hits / float(min(T, num_pos))

        # Normalized DCG@100 with gains 2^rel - 1 and natural log
        dcg = 0.0
        for m, rank in ranks_map.items():
            if rank > max_k:
                continue
            g = rel.get(m, 0.0)
            if g <= 0:
                continue
            dcg += _gain(g) / math.log(rank + 1.0)
        ideal = sorted((_gain(v) for v in rel.values() if v > 0), reverse=True)[:max_k]
        idcg = sum(g / math.log(i + 2.0) for i, g in enumerate(ideal))
        ndcg_i = dcg / idcg if idcg > 0 else 0.0

        # MRR: sum of reciprocal ranks for retrieved positives (only if in top-100)
        finite_ranks = pos_ranks[np.isfinite(pos_ranks)]
        mrr_i = float(np.sum(1.0 / finite_ranks)) if finite_ranks.size > 0 else 0.0

        # MAR: sum of ranks, penalizing missing items with rank 101
        mar_i = float(np.sum(np.where(np.isfinite(pos_ranks), pos_ranks, max_k + 1.0)))

        recall_sum += recall_i
        ndcg_sum += ndcg_i
        mrr_sum += mrr_i
        mar_sum += mar_i
        n_effective += 1

    if n_effective == 0:
        return {
            f"recall@{T}": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "mar": 0.0,
        }

    N = float(n_effective)
    return {
        f"recall@{T}": recall_sum / N,
        "ndcg": ndcg_sum / N,
        "mrr": mrr_sum / N,
        "mar": mar_sum / N,
    }


@dataclass
class Interactions:
    """
    Sparse representation of interactions for one split.

    user_ids[i], item_ids[i], time_ids[i], counts[i] correspond to
    one (user, item, time_bin) cell with non-zero count.
    """

    user_ids: torch.LongTensor  # shape (nnz,)
    item_ids: torch.LongTensor  # shape (nnz,)
    time_ids: torch.LongTensor  # shape (nnz,)
    counts: torch.FloatTensor  # shape (nnz,) # may be long too?


def compute_global_min_max_timestamp(
    dfs: List[pd.DataFrame],
    time_col: int = 3,
) -> Tuple[int, int]:
    """
    Given a list of dataframes, each with an integer Unix timestamp column,
    return the global min and max timestamps.
    """
    min_ts = min(int(df[time_col].min()) for df in dfs)
    max_ts = max(int(df[time_col].max()) for df in dfs)
    return min_ts, max_ts


def make_time_bins(
    dfs: List[pd.DataFrame],
    n_bins: Optional[int] = None,
    bin_size_seconds: Optional[int] = None,
    time_col: int = 3,
) -> np.ndarray:
    """
    Build global time bin edges shared by train/val/test.

    - If n_bins is provided, we use that.
    - Else if bin_size_seconds is provided, we use that.
    - Else we default to 60-day bins (DEFAULT_BIN_SIZE_SECONDS).

    Returns
    -------
    bin_edges : np.ndarray of shape (T+1,)
        Edges such that bin t corresponds to
            [bin_edges[t], bin_edges[t+1])
        and t is in {0, ..., T-1}.
    """
    t_min, t_max = compute_global_min_max_timestamp(dfs, time_col=time_col)

    # Decide binning strategy
    if n_bins is not None and bin_size_seconds is not None:
        raise ValueError("Specify at most one of n_bins or bin_size_seconds.")

    if n_bins is None:
        if bin_size_seconds is None:
            bin_size_seconds = DEFAULT_BIN_SIZE_SECONDS

        span = t_max - t_min
        if bin_size_seconds is None:
            bin_size_seconds = DEFAULT_BIN_SIZE_SECONDS
        n_bins = int(np.ceil(span / bin_size_seconds))
        n_bins = max(n_bins, 1)
        bin_edges = t_min + bin_size_seconds * np.arange(n_bins + 1, dtype=np.int64)

    # +1 so that the last timestamp falls in the last bin
    bin_edges = np.linspace(
        t_min,
        t_max + 1,
        num=n_bins + 1,
        dtype=np.int64,
    )
    return bin_edges


def assign_time_bins(
    timestamps: pd.Series,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Map integer timestamps to bin indices in [0, T-1].

    Uses np.digitize under the hood.
    """
    ts = timestamps.to_numpy(dtype=np.int64)

    # np.digitize with 'bins' of length T-1 (we drop first/last edge for stability)
    # Another convenient trick is to digitize against bin_edges[1:-1]
    # so the result is already in [0, T-1].
    bins = bin_edges[1:-1]  # internal edges only
    t_ids = np.digitize(ts, bins, right=False)  # returns in [0, T-1]
    return t_ids.astype(np.int64)


def make_interactions(
    df: pd.DataFrame,
    bin_edges: np.ndarray,
    user_col: int = 0,
    item_col: int = 1,
    count_col: int = 2,
    time_col: int = 3,
) -> Interactions:
    """
    Turn a (user, item, click, timestamp) dataframe into a sparse
    (user, item, time_bin) -> count representation.

    - Users and items are assumed to already be 0..N-1 and 0..M-1.
    - Column 'count_col' is 0/1, but we still aggregate anyway
      in case multiple events fall into the same time bin.
    """
    df_local = df[[user_col, item_col, count_col, time_col]].copy()
    df_local["t_bin"] = assign_time_bins(df_local[time_col], bin_edges)

    # Group by (user, item, time_bin) and sum counts
    grouped = (
        df_local.groupby([user_col, item_col, "t_bin"], sort=False)[count_col]
        .sum()
        .reset_index()
    )
    grouped[count_col] = (grouped[count_col] > 0).astype(np.float32)

    # Convert to torch tensors
    user_ids = torch.LongTensor(grouped[user_col].to_numpy(dtype=np.int64))
    item_ids = torch.LongTensor(grouped[item_col].to_numpy(dtype=np.int64))
    time_ids = torch.LongTensor(grouped["t_bin"].to_numpy(dtype=np.int64))
    counts = torch.FloatTensor(grouped[count_col].to_numpy(dtype=np.float32))

    return Interactions(
        user_ids=user_ids,
        item_ids=item_ids,
        time_ids=time_ids,
        counts=counts,
    )


def prepare_splits_with_time_bins(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_bins: Optional[int] = None,
    bin_size_seconds: Optional[int] = None,
) -> Tuple[Interactions, Interactions, Interactions, np.ndarray]:
    """
    Main entry point:

    - builds global time bins from train+val+test
    - aggregates each split into sparse Interactions
    - returns (train_int, val_int, test_int, bin_edges)

    If both n_bins and bin_size_seconds are None:
        uses 60-day bins by default.
    """
    dfs = [df_train, df_val, df_test]
    bin_edges = make_time_bins(
        dfs,
        n_bins=n_bins,
        bin_size_seconds=bin_size_seconds,
        time_col=3,
    )

    train_int = make_interactions(df_train, bin_edges)
    val_int = make_interactions(df_val, bin_edges)
    test_int = make_interactions(df_test, bin_edges)

    return train_int, val_int, test_int, bin_edges


# --- Plotting utilities


def plot_training_history(
    history: Dict[str, List[Optional[float]]],
    title: str = "Training curve",
):
    """
    Plot train ELBO and validation predictive log-likelihood over epochs.

    history should be the dict returned by train_dpf.
    """
    epochs = np.array(history["epoch"], dtype=float)
    elbo = np.array(history["elbo"], dtype=float)

    # Filter out epochs where val_pred_ll is None
    val_ll_list = history.get("val_pred_ll", [])

    val_epochs = []
    val_vals = []
    for e, v in zip(epochs, val_ll_list):
        if v is not None:
            val_epochs.append(e)
            val_vals.append(v)

    fig, ax = plt.subplots(figsize=(8, 4))

    elbo = -elbo
    val_vals = -np.array(val_vals, dtype=float)
    ax.plot(epochs, elbo, label="Train ELBO * (-1)")
    if len(val_epochs) > 0:
        ax.plot(
            np.array(val_epochs, dtype=float),
            np.array(val_vals, dtype=float),
            label="Validation predictive NLL",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def _bin_centers_from_edges(bin_edges: np.ndarray) -> np.ndarray:
    """
    Convert bin edges (length T+1) to bin centers (length T) in Unix seconds.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    return edges[:-1] + 0.5 * np.diff(edges)


def _to_datetimes(bin_centers: np.ndarray) -> pd.DatetimeIndex:
    """
    Convert Unix seconds (float) to pandas datetimes.
    """
    return pd.to_datetime(bin_centers, unit="s")


def pick_most_active_user(interactions: Interactions, top_n: int = 1) -> list[int]:
    """
    Heuristic: return IDs of users with most clicks in 'interactions'.
    """
    u = interactions.user_ids.cpu().numpy()
    c = interactions.counts.cpu().numpy()
    max_user = int(u.max()) if u.size > 0 else -1
    if max_user < 0:
        return []

    clicks_per_user = np.bincount(u, weights=c, minlength=max_user + 1)
    order = np.argsort(-clicks_per_user)
    return [int(uid) for uid in order[:top_n]]


def pick_most_active_item(interactions: Interactions, top_n: int = 1) -> list[int]:
    """
    Heuristic: return IDs of items with most clicks in 'interactions'.
    """
    m = interactions.item_ids.cpu().numpy()
    c = interactions.counts.cpu().numpy()
    max_item = int(m.max()) if m.size > 0 else -1
    if max_item < 0:
        return []

    clicks_per_item = np.bincount(m, weights=c, minlength=max_item + 1)
    order = np.argsort(-clicks_per_item)
    return [int(i) for i in order[:top_n]]


def user_factor_trajectory(
    model: DynamicPoissonFactorization,
    user_id: int,
) -> np.ndarray:
    """
    Returns a (T, K) array with factor 'expression' for a given user
    at all time bins.

    Here we use E[theta_{nkt}] = E[exp(ubar_{nk} + u_{nkt})],
    i.e. model.expected_user_factors(...) for all t.
    """
    device = model.device
    T, _ = model.T, model.K

    user_ids = torch.full((T,), user_id, dtype=torch.long, device=device)
    time_ids = torch.arange(T, dtype=torch.long, device=device)

    with torch.no_grad():
        theta = model.expected_user_factors(user_ids, time_ids)  # (T, K) #type:ignore

    return theta.cpu().numpy()  # shape (T, K)


def item_factor_trajectory(
    model: DynamicPoissonFactorization,
    item_id: int,
) -> np.ndarray:
    """
    Returns a (T, K) array with factor 'expression' for a given item
    at all time bins:

      beta_{mkt} = E[exp(vbar_{mk} + v_{mkt})].
    """
    device = model.device
    T, _ = model.T, model.K

    item_ids = torch.full((T,), item_id, dtype=torch.long, device=device)
    time_ids = torch.arange(T, dtype=torch.long, device=device)

    with torch.no_grad():
        beta = model.expected_item_factors(item_ids, time_ids)  # (T, K) #type:ignore

    return beta.cpu().numpy()  # shape (T, K)


def plot_user_evolution(
    model: DynamicPoissonFactorization,
    train_int: Interactions,
    all_int: Interactions,
    bin_edges: np.ndarray,
    user_id: int,
    num_factors: int = 4,
    factor_labels: Optional[Dict[int, str]] = None,
    truncate_to_train: bool = True,
):
    """
    Figure-1–style plot for a single user.

    - Top: total clicks per time bin (train + val + test).
    - Bottom: factor expressions over time (E[theta_{nkt}]).

    If truncate_to_train=True, we only show bins up to the last
    *training* time bin (globally). This avoids plotting unconstrained
    factors in bins where the model never saw data.
    """
    T = model.T
    assert T == len(bin_edges) - 1, "T must match bin_edges"

    # --- Time axis (bin centers -> datetimes) ---
    centers = _bin_centers_from_edges(bin_edges)
    dates = _to_datetimes(centers)

    # --- Click series for this user, using ALL splits ---
    def _user_click_series_all(
        ints: Sequence[Interactions],
        user_id: int,
        T: int,
    ) -> np.ndarray:
        clicks = np.zeros(T, dtype=np.float64)
        for inter in ints:
            u = inter.user_ids.cpu().numpy()
            t = inter.time_ids.cpu().numpy()
            c = inter.counts.cpu().numpy()
            mask = u == user_id
            for ti, ci in zip(t[mask], c[mask]):
                clicks[int(ti)] += float(ci)
        return clicks

    clicks_t_full = _user_click_series_all(
        [
            train_int,
            all_int,
        ],  # 'all_int' already contains val+test; including train again is harmless
        user_id,
        T,
    )

    # --- Factor trajectories over all time bins
    theta_full = user_factor_trajectory(model, user_id)  # (T, K)

    # --- Truncate to training horizon, if requested
    if truncate_to_train and train_int.time_ids.numel() > 0:
        global_last_train_bin = int(train_int.time_ids.max().item())
        t_max = min(global_last_train_bin, T - 1)
    else:
        t_max = T - 1

    sl = slice(0, t_max + 1)

    clicks_t = clicks_t_full[sl]
    theta = theta_full[sl, :]
    dates = dates[sl]

    # --- Choose most-expressed factors for this user
    mean_per_factor = theta.mean(axis=0)
    order = np.argsort(-mean_per_factor)
    chosen_factors = order[:num_factors]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )

    # --- Top: clicks
    ax_top.plot(dates, clicks_t, marker="o")
    ax_top.set_ylabel("Click frequency")
    ax_top.set_title(f"User {user_id} – clicks and factor expression over time")

    # --- Bottom: factors
    for k in chosen_factors:
        label = factor_labels.get(k, f"Factor {k}") if factor_labels else f"Factor {k}"
        ax_bottom.plot(dates, theta[:, k], label=label)

    ax_bottom.set_ylabel("Factor expression (E[theta_{nkt}])")
    ax_bottom.set_xlabel("Time")
    ax_bottom.legend(loc="upper left", frameon=False)

    return fig


def plot_item_evolution(
    model: DynamicPoissonFactorization,
    train_int: Interactions,
    all_int: Interactions,
    bin_edges: np.ndarray,
    item_id: int,
    num_factors: int = 4,
    factor_labels: Optional[Dict[int, str]] = None,
    truncate_to_train: bool = True,
):
    """
    Figure-1–style plot for a single item.

    - Top: total accesses per time bin (train + val + test).
    - Bottom: factor expressions over time (E[beta_{mkt}]).

    If truncate_to_train=True, only show bins up to the last
    training time bin globally.
    """
    T = model.T
    assert T == len(bin_edges) - 1, "T must match bin_edges"

    centers = _bin_centers_from_edges(bin_edges)
    dates = _to_datetimes(centers)

    # --- Access series for this item across all splits
    def _item_click_series_all(
        ints: Sequence[Interactions],
        item_id: int,
        T: int,
    ) -> np.ndarray:
        clicks = np.zeros(T, dtype=np.float64)
        for inter in ints:
            m = inter.item_ids.cpu().numpy()
            t = inter.time_ids.cpu().numpy()
            c = inter.counts.cpu().numpy()
            mask = m == item_id
            for ti, ci in zip(t[mask], c[mask]):
                clicks[int(ti)] += float(ci)
        return clicks

    clicks_t_full = _item_click_series_all(
        [train_int, all_int],
        item_id,
        T,
    )

    beta_full = item_factor_trajectory(model, item_id)  # (T, K)

    if truncate_to_train and train_int.time_ids.numel() > 0:
        global_last_train_bin = int(train_int.time_ids.max().item())
        t_max = min(global_last_train_bin, T - 1)
    else:
        t_max = T - 1

    sl = slice(0, t_max + 1)

    clicks_t = clicks_t_full[sl]
    beta = beta_full[sl, :]
    dates = dates[sl]

    mean_per_factor = beta.mean(axis=0)
    order = np.argsort(-mean_per_factor)
    chosen_factors = order[:num_factors]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )

    ax_top.plot(dates, clicks_t, marker="o")
    ax_top.set_ylabel("Access frequency")
    ax_top.set_title(f"Item {item_id} – accesses and factor expression over time")

    for k in chosen_factors:
        label = factor_labels.get(k, f"Factor {k}") if factor_labels else f"Factor {k}"
        ax_bottom.plot(dates, beta[:, k], label=label)

    ax_bottom.set_ylabel("Factor expression (E[beta_{mkt}])")
    ax_bottom.set_xlabel("Time")
    ax_bottom.legend(loc="upper left", frameon=False)

    return fig


def plot_global_factor_popularity(
    model: DynamicPoissonFactorization,
    use_items: bool = True,
    normalize_per_time: bool = True,
    factor_labels: Optional[Dict[int, str]] = None,
    max_time: Optional[int] = None,
):
    """
    Figure-3–style plot: evolution of factors over time.

    max_time: if not None, only use time bins [0, max_time) for the plot.
              A good choice is max_time = last training time bin + 1.
    """
    T, K = model.T, model.K

    if max_time is None:
        Tmax = T
    else:
        Tmax = min(max_time, T)

    with torch.no_grad():
        if use_items:
            mu_dyn = model.mu_v[:, :, :Tmax]  # (M, K, Tmax)
            mu_static = model.mu_v_bar[:, :, None]  # (M, K, 1)
            var_dyn = torch.exp(model.logvar_v[:, :, :Tmax])
            var_static = torch.exp(model.logvar_v_bar)[:, :, None]

            mu_total = mu_dyn + mu_static
            var_total = var_dyn + var_static

            beta = torch.exp(mu_total + 0.5 * var_total)  # (M, K, Tmax)
            f_k_t = beta.sum(dim=0)  # (K, Tmax)
        else:
            mu_dyn = model.mu_u[:, :, :Tmax]
            mu_static = model.mu_u_bar[:, :, None]
            var_dyn = torch.exp(model.logvar_u[:, :, :Tmax])
            var_static = torch.exp(model.logvar_u_bar)[:, :, None]

            mu_total = mu_dyn + mu_static
            var_total = var_dyn + var_static

            theta = torch.exp(mu_total + 0.5 * var_total)  # (N, K, Tmax)
            f_k_t = theta.sum(dim=0)  # (K, Tmax)

    f_t_k = f_k_t.transpose(0, 1).cpu().numpy()  # (Tmax, K)

    if normalize_per_time:
        denom = f_t_k.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        f_t_k = f_t_k / denom

    xs = np.arange(Tmax)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(K):
        label = factor_labels.get(k, f"Factor {k}") if factor_labels else f"Factor {k}"
        ax.plot(xs, f_t_k[:, k], label=label)

    ax.set_xlabel("Time bin index")
    ax.set_ylabel(
        "Relative popularity (per time)"
        if normalize_per_time
        else "Total factor intensity"
    )
    side = "items" if use_items else "users"
    ax.set_title(f"Global factor evolution over time ({side})")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize="small",
    )
    fig.tight_layout()

    return fig


def plot_item_static_factors(
    model: DynamicPoissonFactorization,
    item_id: int,
    top_k: Optional[int] = None,
    factor_labels: Optional[Dict[int, str]] = None,
):
    """
    Figure-4–style plot: static factors for a single item.

    We use the static log-factors v_bar[m, k] and their variance:

        E[beta_bar_{mk}] = E[exp(v_bar_{mk})]
                         = exp(mu + 0.5 * var)

    This is time-independent; we show it as a bar chart.
    """
    with torch.no_grad():
        mu = model.mu_v_bar[item_id, :]  # (K,)
        var = torch.exp(model.logvar_v_bar[item_id, :])  # (K,)
        beta_bar = torch.exp(mu + 0.5 * var).cpu().numpy()  # (K,)

    K = model.K
    idx = np.arange(K)

    if top_k is not None and top_k < K:
        order = np.argsort(-beta_bar)[:top_k]
        idx = order
        vals = beta_bar[order]
        labels = [
            factor_labels.get(k, f"Factor {k}") if factor_labels else f"F{k}"
            for k in order
        ]
    else:
        vals = beta_bar
        labels = [
            factor_labels.get(k, f"Factor {k}") if factor_labels else f"F{k}"
            for k in range(K)
        ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(idx)), vals, tick_label=labels)
    ax.set_ylabel("Static factor expression (E[betā_{mk}])")
    ax.set_title(f"Static factors for item {item_id}")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    return fig


def concat_interactions(ints: Sequence[Interactions]) -> Interactions:
    """
    Concatenate multiple Interactions objects.
    Assumes splits are disjoint in time (true for your dataset).
    """
    user_ids = torch.cat([x.user_ids for x in ints])
    item_ids = torch.cat([x.item_ids for x in ints])
    time_ids = torch.cat([x.time_ids for x in ints])
    counts = torch.cat([x.counts for x in ints])

    return Interactions(
        user_ids=user_ids,  # type:ignore
        item_ids=item_ids,  # type:ignore
        time_ids=time_ids,  # type:ignore
        counts=counts,  # type:ignore
    )


if __name__ == "__main__":
    dfs = {}
    for split in ["train", "validation", "test"]:
        dfs[split] = pd.read_csv(f"data/{split}.tsv", sep="\t", header=None)

    df_test_users = pd.read_csv("data/test_users.tsv", sep="\t", header=None)

    # (optional) restrict to test users:
    # for split in ["train", "validation", "test"]:
    #     dfs[split] = filter_to_users(dfs[split], df_test_users)

    # Build time bins + sparse Interactions
    train_int, val_int, test_int, bin_edges = prepare_splits_with_time_bins(
        dfs["train"],
        dfs["validation"],
        dfs["test"],
    )

    # all_int used for click frequencies, (so last bins include val/test clicks)
    all_int = concat_interactions([train_int, val_int, test_int])

    # global last training bin index (inclusive)
    global_last_train_bin = int(train_int.time_ids.max().item())

    # Infer problem dimensions
    T = len(bin_edges) - 1  # number of time bins
    num_users, num_items = infer_num_users_items_from_interactions(
        train_int, val_int, test_int
    )

    print(f"#users={num_users}, #items={num_items}, #time_bins={T}")

    # Create DPF model with default hyperparameters
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    hyper = DPFHyperParams(
        K=5,
        prior_var_dynamic=10.0,
        prior_var_static=10.0,
    )
    model = DynamicPoissonFactorization(
        num_users=num_users,
        num_items=num_items,
        num_times=T,
        hyper=hyper,
        device=device,
    ).to(device)

    print("device:", device)
    print(model)

    # Train model
    history = train_dpf(
        model,
        train_int,
        val_int=val_int,
        num_epochs=10000,
        lr=1e-2,
        optimizer_name="adamw",
        verbose=True,
        min_epochs_before_stop=1000,
    )

    # Paper metrics (same as paper/code): Recall@50, NDCG, MRR, MAR
    # k=5, T=50
    test_users_np = df_test_users[0].to_numpy(dtype=np.int64)
    paper_metrics = paper_ranking_metrics(
        model,
        train_int,
        val_int,
        test_int,
        T=50,
        test_users=test_users_np,
    )
    print(
        f"Paper metrics -> Recall@50 = {paper_metrics['recall@50']:.4f}, "
        # f"NDCG = {paper_metrics['ndcg']:.4f}, "
        # f"MRR = {paper_metrics['mrr']:.4f}, "
        # f"MAR = {paper_metrics['mar']:.4f}"
    )
    assets_dir = os.path.join("assets")
    os.makedirs(assets_dir, exist_ok=True)

    # --- Example: pick top-N active users + items
    PLOT_N_USERS = 1
    PLOT_N_ITEMS = 1
    PLOT_CHOOSE_RANDOM = True

    if PLOT_CHOOSE_RANDOM:
        # Pick random users/items
        rng = np.random.default_rng()
        user_pool = np.arange(num_users)
        if user_pool.size == 0:
            active_users = []
        else:
            k_u = min(PLOT_N_USERS, user_pool.size)
            active_users = rng.choice(user_pool, size=k_u, replace=False).tolist()
        item_pool = np.arange(num_items)
        if item_pool.size == 0:
            active_items = []
        else:
            k_m = min(PLOT_N_ITEMS, item_pool.size)
            active_items = rng.choice(item_pool, size=k_m, replace=False).tolist()
    else:
        # Pick most active users/items from training set
        active_users = pick_most_active_user(train_int, top_n=PLOT_N_USERS)
        active_items = pick_most_active_item(train_int, top_n=PLOT_N_ITEMS)

    # Build combined interactions for plotting clicks
    all_int = concat_interactions([train_int, val_int, test_int])
    max_time_train = int(train_int.time_ids.max().item()) + 1

    # Generate and save user evolution plots for each of the top users
    for u in active_users:
        fig_user = plot_user_evolution(
            model,
            train_int=train_int,
            all_int=all_int,
            bin_edges=bin_edges,
            user_id=u,
            num_factors=5,
            truncate_to_train=True,
        )
        fig_user.suptitle(f"User {u} – evolution", y=1.02)
        user_fname = os.path.join(assets_dir, f"user_{u}_evolution.pdf")
        fig_user.savefig(user_fname)
        print(f"Saved user evolution plot to {user_fname}")
        plt.close(fig_user)

    # Generate and save item evolution + static factor plots for each of the top items
    for m in active_items:
        fig_item = plot_item_evolution(
            model,
            train_int=train_int,
            all_int=all_int,
            bin_edges=bin_edges,
            item_id=m,
            num_factors=4,
            truncate_to_train=True,
        )
        fig_item.suptitle(f"Item {m} – evolution", y=1.02)
        item_fname = os.path.join(assets_dir, f"item_{m}_evolution.pdf")
        fig_item.savefig(item_fname)
        print(f"Saved item evolution plot to {item_fname}")
        plt.close(fig_item)

        fig_item_static = plot_item_static_factors(
            model,
            item_id=m,
            top_k=5,
        )
        item_static_fname = os.path.join(assets_dir, f"item_{m}_static_factors.pdf")
        fig_item_static.savefig(item_static_fname)
        print(f"Saved item static factors plot to {item_static_fname}")
        plt.close(fig_item_static)

    # --- Global factor evolution ( Figure 3)
    fig_global = plot_global_factor_popularity(
        model,
        use_items=True,
        normalize_per_time=True,
        max_time=max_time_train,
    )

    global_fname = os.path.join(assets_dir, "global_factor_evolution.pdf")
    fig_global.savefig(global_fname)
    print(f"Saved global factor evolution plot to {global_fname}")

    # --- Training curve: ELBO & validation predictive LL over epochs
    fig_train = plot_training_history(
        history,
        title="ELBO and validation predictive log-likelihood over epochs",
    )
    train_curve_fname = os.path.join(assets_dir, "training_elbo_val_pred_ll.pdf")
    fig_train.savefig(train_curve_fname)
    print(f"Saved training curve plot to {train_curve_fname}")
    plt.close(fig_train)

    plt.close("all")
