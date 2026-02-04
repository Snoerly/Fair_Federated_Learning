"""server/aggregation.py

AFW - Adaptive Fairness-weighted Aggregation (called AFA/AFW in the paper).

Paper idea:
1) Measure each client's group fairness bias Fi
2) Compute aggregation coefficients alpha_i = Fi / sum(Fi)
3) Aggregate client models using alpha_i

Important note:
- The paper's text uses an additive update form:
  w_global^{t+1} = w_global^{t} + sum_i alpha_i * w_i^{t+1}
  In standard FL we usually set:
  w_global^{t+1} = sum_i alpha_i * w_i^{t+1}

Here we implement the *standard weighted aggregation* form, which is
what you want if local models already started from the same global model
in each round. If you want the additive form, set use_additive=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AFWResult:
    weights: Dict[int, float]  # alpha_i


def compute_alphas(fairness_bias: Dict[int, float], eps: float = 1e-12) -> Dict[int, float]:
    """Compute alpha_i = Fi / sum(Fi).

    If all Fi are 0 (already perfectly fair), fallback to uniform weights.
    """
    total = float(sum(max(0.0, v) for v in fairness_bias.values()))
    if total <= eps:
        n = len(fairness_bias)
        return {cid: 1.0 / n for cid in fairness_bias}
    return {cid: float(max(0.0, Fi) / total) for cid, Fi in fairness_bias.items()}


def aggregate_state_dicts(
    client_state_dicts: Dict[int, Dict[str, "np.ndarray | object"]],
    alphas: Dict[int, float],
    use_additive: bool = False,
    global_state_dict: Dict[str, "np.ndarray | object"] | None = None,
):
    """Aggregate model parameters.

    Parameters
    ----------
    client_state_dicts:
        {client_id: state_dict}
        state_dict values can be numpy arrays or torch tensors.
    alphas:
        {client_id: alpha}
    use_additive:
        If True, apply w_global = w_global + sum_i alpha_i * w_i
        Else (default), w_global = sum_i alpha_i * w_i
    global_state_dict:
        Required if use_additive=True.

    Returns
    -------
    new_state_dict: aggregated state_dict in the same type as inputs
    """
    client_ids = list(client_state_dicts.keys())
    if set(client_ids) != set(alphas.keys()):
        raise ValueError("client_state_dicts and alphas must have same client IDs")

    # Helper to convert tensors to numpy for aggregation
    def to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Determine keys
    keys = list(next(iter(client_state_dicts.values())).keys())

    # Aggregate in numpy
    agg_np = {}
    for k in keys:
        s = None
        for cid in client_ids:
            w = to_np(client_state_dicts[cid][k])
            a = alphas[cid]
            s = w * a if s is None else s + w * a
        agg_np[k] = s

    if use_additive:
        if global_state_dict is None:
            raise ValueError("global_state_dict is required for additive aggregation")
        for k in keys:
            agg_np[k] = to_np(global_state_dict[k]) + agg_np[k]

    # Convert back to original types (torch tensors if the first client uses torch)
    first_val = next(iter(client_state_dicts.values()))[keys[0]]
    if hasattr(first_val, "detach"):
        import torch
        new_state = {k: torch.tensor(v, dtype=first_val.dtype, device=first_val.device) for k, v in agg_np.items()}
    else:
        new_state = agg_np

    return new_state
