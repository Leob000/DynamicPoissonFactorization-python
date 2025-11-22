from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .data import Interactions, last_training_timestep
from .model import DynamicPoissonFactorization


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
