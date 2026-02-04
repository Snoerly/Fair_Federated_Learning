"""server/gbcfss.py

GBCFSS - Greedy-Based Clean and Fair Sample Selection

The paper references GBCFSS as a greedy selection algorithm that scores
samples by (a) cleanliness and (b) fairness contribution, and selects a
subset while keeping constraints satisfied.

The original Algorithm 1 in the paper's preliminaries is presented in a
somewhat generic form (and appears adapted from prior work).

This implementation provides a practical, paper-aligned version:
- Cleanliness score: higher if model loss is lower (cleaner sample)
- Fairness contribution: higher if sample helps reduce EO or DP gap

We implement a greedy selector that:
1) scores each sample
2) sorts samples by score descending
3) selects up to a budget while optionally enforcing a group-balance
   constraint to prevent extreme skew.

You can tune:
- cleanliness_weight
- fairness_weight
- group_balance (optional)

Inputs:
- X: np.ndarray [N, d]
- y: np.ndarray [N]
- g: np.ndarray [N]
- y_score: np.ndarray [N] predicted probability for y=1 (from current global model)

Outputs:
- selected indices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GBCFSSConfig:
    budget: int
    cleanliness_weight: float = 1.0
    fairness_weight: float = 1.0
    group_balance: Optional[float] = None
    """If set, enforce that each group has at least group_balance fraction
    of selected samples. Example: 0.2 means >=20% from each group.
    Only meaningful for binary groups.
    """


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0 - eps))


def binary_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Per-sample logistic loss for y in {0,1} and score in [0,1]."""
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    return -(y_true * _safe_log(y_score) + (1.0 - y_true) * _safe_log(1.0 - y_score))


def fairness_contribution_scores(
    y_true: np.ndarray,
    g: np.ndarray,
    mode: str = "eo",
) -> np.ndarray:
    """Compute a per-sample *heuristic* fairness contribution score.

    This approximates whether selecting the sample tends to reduce the
    disparity between groups.

    - For EO: focus only on true positives (y=1). Boost samples from the
      disadvantaged group among positives.
    - For DP: boost samples from the disadvantaged group overall.

    We define disadvantaged group as the one with *lower* prevalence of
    the relevant condition.

    Returns
    -------
    s_fair : np.ndarray [N] in [0,1] (higher is better for fairness)
    """
    y_true = y_true.reshape(-1)
    g = g.reshape(-1)

    # default neutral
    s = np.full_like(y_true, 0.5, dtype=np.float64)

    if mode == "eo":
        pos = (y_true == 1)
        if not np.any(pos):
            return s

        # Among positives, find which group has fewer samples
        n0 = np.sum((g == 0) & pos)
        n1 = np.sum((g == 1) & pos)
        disadvantaged = 0 if n0 < n1 else 1

        # Give higher score to positives from disadvantaged group
        s[pos & (g == disadvantaged)] = 1.0
        s[pos & (g != disadvantaged)] = 0.0
        return s

    if mode == "dp":
        # Overall group imbalance
        n0 = np.sum(g == 0)
        n1 = np.sum(g == 1)
        disadvantaged = 0 if n0 < n1 else 1

        s[(g == disadvantaged)] = 1.0
        s[(g != disadvantaged)] = 0.0
        return s

    raise ValueError(f"Unknown mode: {mode}")


def greedy_select_gbcfss(
    y_true: np.ndarray,
    g: np.ndarray,
    y_score: np.ndarray,
    cfg: GBCFSSConfig,
    fairness_mode: str = "eo",
) -> np.ndarray:
    """Greedy selection of a clean & fair subset.

    Cleanliness: prefer low-loss samples.
    Fairness: prefer samples that help balance groups under EO/DP.

    Score_i = w_clean * clean_i + w_fair * fair_i

    clean_i is normalized to [0,1] where 1 means very clean (low loss).
    fair_i is in {0,1} (or [0,1]) from heuristic.

    Returns
    -------
    selected_idx : np.ndarray of selected indices
    """
    y_true = y_true.reshape(-1)
    g = g.reshape(-1)
    y_score = y_score.reshape(-1)

    N = y_true.shape[0]
    budget = min(cfg.budget, N)

    # 1) Cleanliness score
    loss = binary_log_loss(y_true, y_score)

    # Convert loss -> cleanliness in [0,1] (lower loss => higher cleanliness)
    # Use robust scaling via percentiles
    lo, hi = np.percentile(loss, [5, 95])
    denom = max(hi - lo, 1e-12)
    clean = 1.0 - np.clip((loss - lo) / denom, 0.0, 1.0)

    # 2) Fairness contribution score
    fair = fairness_contribution_scores(y_true, g, mode=fairness_mode)

    # 3) Combined score
    score = cfg.cleanliness_weight * clean + cfg.fairness_weight * fair

    # 4) Greedy pick
    order = np.argsort(-score)
    selected: List[int] = []

    if cfg.group_balance is None:
        selected = order[:budget].tolist()
        return np.array(selected, dtype=int)

    # Group-balance constrained selection (binary groups)
    min_per_group = int(np.floor(cfg.group_balance * budget))
    count = {0: 0, 1: 0}

    for idx in order:
        if len(selected) >= budget:
            break

        gi = int(g[idx])

        # If a group has not met its minimum, prioritize it
        # Otherwise accept normally
        if count[gi] < min_per_group:
            selected.append(int(idx))
            count[gi] += 1
            continue

        # If both minimums met, fill remaining freely
        if count[0] >= min_per_group and count[1] >= min_per_group:
            selected.append(int(idx))
            count[gi] += 1
            continue

        # If other group still below min, skip this one for now
        other = 1 - gi
        if count[other] < min_per_group:
            continue

        selected.append(int(idx))
        count[gi] += 1

    # If not enough due to constraints, fill remaining best scores
    if len(selected) < budget:
        remaining = [int(i) for i in order if int(i) not in set(selected)]
        selected.extend(remaining[: budget - len(selected)])

    return np.array(selected, dtype=int)
