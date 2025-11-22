from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import Interactions, concat_interactions
from .model import DynamicPoissonFactorization


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


def choose_plot_entities(
    num_users: int,
    num_items: int,
    train_int: Interactions,
    n_users: int,
    n_items: int,
    choose_random: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[int]]:
    """
    Pick which users/items to visualize, either randomly or by activity.
    """
    if rng is None:
        rng = np.random.default_rng()

    if choose_random:
        user_pool = np.arange(num_users)
        if user_pool.size == 0:
            active_users: list[int] = []
        else:
            k_u = min(n_users, user_pool.size)
            active_users = rng.choice(user_pool, size=k_u, replace=False).tolist()

        item_pool = np.arange(num_items)
        if item_pool.size == 0:
            active_items: list[int] = []
        else:
            k_m = min(n_items, item_pool.size)
            active_items = rng.choice(item_pool, size=k_m, replace=False).tolist()
    else:
        active_users = pick_most_active_user(train_int, top_n=n_users)
        active_items = pick_most_active_item(train_int, top_n=n_items)

    return active_users, active_items


def generate_plots(
    model: DynamicPoissonFactorization,
    train_int: Interactions,
    val_int: Interactions,
    test_int: Interactions,
    bin_edges: np.ndarray,
    history: Dict[str, List[Optional[float]]],
    assets_dir: str,
    plot_n_users: int = 1,
    plot_n_items: int = 1,
    choose_random: bool = True,
):
    """
    Orchestrate user/item/global plots and save them under assets_dir.
    """
    os.makedirs(assets_dir, exist_ok=True)
    all_int = concat_interactions([train_int, val_int, test_int])

    max_time_train = (
        int(train_int.time_ids.max().item()) + 1
        if train_int.time_ids.numel() > 0
        else model.T
    )

    active_users, active_items = choose_plot_entities(
        num_users=model.N,
        num_items=model.M,
        train_int=train_int,
        n_users=plot_n_users,
        n_items=plot_n_items,
        choose_random=choose_random,
    )

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

    fig_global = plot_global_factor_popularity(
        model,
        use_items=True,
        normalize_per_time=True,
        max_time=max_time_train,
    )

    global_fname = os.path.join(assets_dir, "global_factor_evolution.pdf")
    fig_global.savefig(global_fname)
    print(f"Saved global factor evolution plot to {global_fname}")

    fig_train = plot_training_history(
        history,
        title="ELBO and validation predictive log-likelihood over epochs",
    )
    train_curve_fname = os.path.join(assets_dir, "training_elbo_val_pred_ll.pdf")
    fig_train.savefig(train_curve_fname)
    print(f"Saved training curve plot to {train_curve_fname}")
    plt.close(fig_train)

    plt.close("all")
