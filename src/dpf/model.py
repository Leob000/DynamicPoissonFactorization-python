from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .data import Interactions


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
