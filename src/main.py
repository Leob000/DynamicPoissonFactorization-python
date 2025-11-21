from __future__ import annotations
from collections import defaultdict

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 60 days in seconds
DEFAULT_BIN_SIZE_SECONDS = 60 * 24 * 3600


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

        # E[lam_{nmt}] = sum_k E[theta_{nkt} beta_{mkt}]
        lam = self.poisson_rate(user_ids, item_ids, time_ids)  # (nnz,) # type:ignore

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

        # phi-optimized lower bound: y * log_lam_tilde - E[lam] - log(y!)
        log_lik = (y * log_lam_tilde - lam - torch.lgamma(y + 1.0)).sum()
        return log_lik

    def responsibilities_phi(
        self,
        interactions: Interactions,
    ) -> torch.Tensor:
        """
        Compute phi_{nmtk} prop= exp(E[log theta_{nkt}] + E[log beta_{mkt}])
        for the provided interactions. Returns a tensor of shape (nnz, K).
        """
        user_ids = interactions.user_ids.to(self.device).long()
        item_ids = interactions.item_ids.to(self.device).long()
        time_ids = interactions.time_ids.to(self.device).long()

        log_theta = self.expected_log_user_factors(
            user_ids,  # type:ignore
            time_ids,  # type:ignore
        )  # (nnz, K)
        log_beta = self.expected_log_item_factors(
            item_ids,  # type:ignore
            time_ids,  # type:ignore
        )  # (nnz, K)
        logits = log_theta + log_beta  # (nnz, K)

        # Softmax gives normalized phi, but we typically don't let grads
        # flow through phi when thinking in coordinate-ascent terms.
        phi = torch.softmax(logits.detach(), dim=-1)
        return phi

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
    num_epochs: int = 100,
    lr: float = 1e-2,
    optimizer_name: str = "lbfgs",
    verbose: bool = True,
) -> None:
    """
    Simple full-batch training loop on 'train_int'.

    - optimizer_name: "adamw" or "lbfgs"
    - full dataset is used per step (possible to add mini-batching?).
    """
    model.train()
    params = list(model.parameters())

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            elbo = model.elbo(train_int)
            loss = -elbo
            loss.backward()
            optimizer.step()

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"[AdamW] epoch={epoch:03d}  ELBO={elbo.item():.3e}")

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

            if verbose and (epoch % 1 == 0 or epoch == num_epochs - 1):
                with torch.no_grad():
                    elbo = model.elbo(train_int)
                print(
                    f"[LBFGS] epoch={epoch:03d}  loss={loss.item():.3e}  ELBO={elbo.item():.3e}"
                )
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")


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


def recall_ndcg_at_k(
    model: DynamicPoissonFactorization,
    test_int: Interactions,
    k: int = 50,
    test_users: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Evaluate Recall@K and NDCG@K over all (user, time_bin) pairs
    that have at least one positive item.

    If test_users is provided (1D np.array of user IDs), restrict to those users.
    """
    import numpy as np

    ut_pos = build_user_time_positives(test_int)

    recalls = []
    ndcgs = []

    for (u, t), pos_items in ut_pos.items():
        if test_users is not None and u not in test_users:
            continue
        if not pos_items:
            continue

        scores = scores_for_user_time(model, u, t)
        topk = torch.topk(scores, k).indices.numpy().tolist()

        hits = [1.0 if item in pos_items else 0.0 for item in topk]
        num_pos = len(pos_items)

        # Recall@K
        recall = sum(hits) / num_pos
        recalls.append(recall)

        # NDCG@K
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = [1.0] * min(num_pos, k)
        idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    if not recalls:
        return 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(ndcgs))


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
        n_bins = int(np.ceil(span / bin_size_seconds))
        n_bins = max(n_bins, 1)

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


def filter_to_users(
    df: pd.DataFrame,
    df_users: pd.DataFrame,
    user_col: int = 0,
) -> pd.DataFrame:
    """
    Keep only rows whose user is in df_users[user_col].
    """
    keep_users = df_users[user_col].to_numpy()
    mask = df[user_col].isin(keep_users)
    return df.loc[mask].reset_index(drop=True)


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

    # Train model (example: small number of epochs to test everything works)
    train_dpf(
        model,
        train_int,
        num_epochs=500,
        lr=1e-2,
        optimizer_name="adamw",
        verbose=True,
    )

    test_users_np = df_test_users[0].to_numpy(dtype=np.int64)
    recall, ndcg = recall_ndcg_at_k(model, test_int, k=100, test_users=test_users_np)
    print(f"Test Recall@50 = {recall:.4f}, NDCG@50 = {ndcg:.4f}")
