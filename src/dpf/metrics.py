from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import torch

from .data import Interactions, last_training_timestep
from .model import DynamicPoissonFactorization


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
