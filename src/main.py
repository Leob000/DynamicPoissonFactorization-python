from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# 60 days in seconds
DEFAULT_BIN_SIZE_SECONDS = 60 * 24 * 3600


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
        # dfs[split].columns = ["user", "item", "rating", "time"]
        # dfs[split]["time_utc"] = pd.to_datetime(dfs[split]["time"], unit="s", utc=True)
    df_test_users = pd.read_csv("data/test_users.tsv", sep="\t", header=None)

    # (optional) restrict to test users:
    # for split in ["train", "validation", "test"]:
    #     dfs[split] = filter_to_users(dfs[split], df_test_users)

    train_int, val_int, test_int, bin_edges = prepare_splits_with_time_bins(
        dfs["train"],
        dfs["validation"],
        dfs["test"],
    )

    T = len(bin_edges) - 1  # number of time bins
    print(T, train_int.user_ids.shape, val_int.user_ids.shape, test_int.user_ids.shape)
