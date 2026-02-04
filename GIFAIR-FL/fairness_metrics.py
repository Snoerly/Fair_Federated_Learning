#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np


def _safe_div(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def compute_fairness_metrics(y_true, y_pred, groups):
    """Compute Accuracy, Equal Opportunity (EOp) gap and Disparate Impact (DI).

    Args:
        y_true: array-like of shape (n_samples,), ground truth labels (0/1).
        y_pred: array-like of shape (n_samples,), predicted labels (0/1).
        groups: array-like of shape (n_samples,), sensitive attribute (0/1),
            where 1 denotes the privileged group and 0 the unprivileged group.

    Returns:
        dict with keys: 'accuracy', 'eop_gap', 'di_ratio'.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups).astype(int)

    # Overall accuracy
    accuracy = (y_true == y_pred).mean() if y_true.size > 0 else 0.0

    # Equal Opportunity: difference in TPR between groups
    # TPR = P(\hat{Y}=1 | Y=1, A=a)
    mask_pos = y_true == 1
    mask_priv = groups == 1
    mask_unpriv = groups == 0

    tpr_priv = _safe_div(
        ((y_pred == 1) & mask_pos & mask_priv).sum(),
        (mask_pos & mask_priv).sum(),
        default=0.0,
    )
    tpr_unpriv = _safe_div(
        ((y_pred == 1) & mask_pos & mask_unpriv).sum(),
        (mask_pos & mask_unpriv).sum(),
        default=0.0,
    )
    eop_gap = abs(tpr_priv - tpr_unpriv)

    # Disparate Impact: ratio of positive prediction rates
    # DI = P(\hat{Y}=1 | A=0) / P(\hat{Y}=1 | A=1)
    p_pos_unpriv = _safe_div(
        ((y_pred == 1) & mask_unpriv).sum(),
        mask_unpriv.sum(),
        default=0.0,
    )
    p_pos_priv = _safe_div(
        ((y_pred == 1) & mask_priv).sum(),
        mask_priv.sum(),
        default=0.0,
    )
    di_ratio = _safe_div(p_pos_unpriv, p_pos_priv, default=0.0)

    return {
        'accuracy': float(accuracy),
        'eop_gap': float(eop_gap),
        'di_ratio': float(di_ratio),
        'tpr_priv': float(tpr_priv),
        'tpr_unpriv': float(tpr_unpriv),
        'p_pos_unpriv': float(p_pos_unpriv),
        'p_pos_priv': float(p_pos_priv),
    }
