from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from dpf import (
    DPFHyperParams,
    DynamicPoissonFactorization,
    generate_plots,
    infer_num_users_items_from_interactions,
    paper_ranking_metrics,
    prepare_splits_with_time_bins,
    train_dpf,
)


def load_splits() -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load train/validation/test TSVs plus the list of test users.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        dfs[split] = pd.read_csv(f"data/{split}.tsv", sep="\t", header=None)

    df_test_users = pd.read_csv("data/test_users.tsv", sep="\t", header=None)
    return dfs, df_test_users


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    num_users: int,
    num_items: int,
    num_times: int,
    device: torch.device,
) -> DynamicPoissonFactorization:
    hyper = DPFHyperParams(
        K=5,
        prior_var_dynamic=10.0,
        prior_var_static=10.0,
    )
    model = DynamicPoissonFactorization(
        num_users=num_users,
        num_items=num_items,
        num_times=num_times,
        hyper=hyper,
        device=device,
    ).to(device)
    return model


def main():
    dfs, df_test_users = load_splits()

    # (optional) restrict to test users:
    # for split in ["train", "validation", "test"]:
    #     dfs[split] = filter_to_users(dfs[split], df_test_users)

    train_int, val_int, test_int, bin_edges = prepare_splits_with_time_bins(
        dfs["train"],
        dfs["validation"],
        dfs["test"],
    )

    T = len(bin_edges) - 1  # number of time bins
    num_users, num_items = infer_num_users_items_from_interactions(
        train_int, val_int, test_int
    )

    print(f"#users={num_users}, #items={num_items}, #time_bins={T}")

    device = choose_device()
    model = build_model(
        num_users=num_users,
        num_items=num_items,
        num_times=T,
        device=device,
    )

    print("device:", device)
    print(model)

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
    generate_plots(
        model,
        train_int=train_int,
        val_int=val_int,
        test_int=test_int,
        bin_edges=bin_edges,
        history=history,
        assets_dir=assets_dir,
        plot_n_users=1,
        plot_n_items=1,
        choose_random=True,
    )


if __name__ == "__main__":
    main()
