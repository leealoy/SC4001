"""Shared evaluation utilities for every metadata and multimodal model.

All models write predictions on the log1p-price scale and call these functions;
this prevents small metric differences between members' experiment scripts.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def metrics(y_true_log, y_pred_log):
    """Return the project's predeclared metrics.

    MAE is calculated on log1p price.  The remaining metrics are calculated in
    USD after clipping predictions at zero dollars.  Inputs must be finite and
    aligned; silently dropping different rows for different models would break
    paired comparisons.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    if y_true_log.shape != y_pred_log.shape:
        raise ValueError("true and predicted arrays have different shapes")
    if not (np.isfinite(y_true_log).all() and np.isfinite(y_pred_log).all()):
        raise ValueError("metrics received missing or non-finite values")

    true_usd = np.expm1(y_true_log)
    pred_usd = np.expm1(np.clip(y_pred_log, 0, None))
    if np.any(true_usd <= 0):
        raise ValueError("all true prices must be greater than zero")

    # Spearman is undefined for a single row or a constant true/predicted
    # vector. This commonly occurs in per-set slices for median baselines; mark
    # it missing explicitly instead of emitting scipy's ConstantInputWarning.
    if (
        len(y_true_log) < 2
        or np.ptp(y_true_log) == 0
        or np.ptp(y_pred_log) == 0
    ):
        rho = np.nan
    else:
        rho = spearmanr(y_true_log, y_pred_log).correlation
    ratio = pred_usd / true_usd
    return dict(
        mae_log=float(np.mean(np.abs(y_true_log - y_pred_log))),
        spearman=float(rho) if np.isfinite(rho) else np.nan,
        medape=float(np.median(np.abs(pred_usd - true_usd) / true_usd)),
        within_2x=float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    )

