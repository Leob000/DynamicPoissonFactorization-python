from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# ~6 months in seconds (matches C++ default 6*30 days)
DEFAULT_BIN_SIZE_SECONDS = 6 * 30 * 24 * 3600


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
