"""Thin wrapper around the vendored TSB-AD evaluation code.

Computes the TSB-AD headline metrics we report: VUS-PR (primary), VUS-ROC,
AUC-PR, AUC-ROC, plus point-level F1. For score-based detectors PointF1 uses
the oracle (best) threshold, as in TSB-AD; for detectors with native binary
decisions (GuardGP routing, SPOT) we additionally compute P/R/F1 from the
actual predictions.

The sliding window is determined from the data via find_length_rank(rank=1),
exactly as in the TSB-AD benchmark scripts.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsb_basic_metrics import basic_metricor, generate_curve
from tsb_slidingwindows import find_length_rank


def sliding_window_of(y: np.ndarray) -> int:
    return int(find_length_rank(np.asarray(y, float).reshape(-1, 1), rank=1))


def eval_scores(labels: np.ndarray, score: np.ndarray, sliding_window: int) -> dict:
    """Threshold-independent metrics from a continuous anomaly score."""
    labels = np.asarray(labels).astype(int)
    score = np.asarray(score, float)
    # guard: constant scores break sklearn/VUS
    if np.all(score == score[0]):
        score = score + 1e-9 * np.arange(len(score))
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    grader = basic_metricor()
    out = {
        "AUC_PR": float(grader.metric_PR(labels, score)),
        "AUC_ROC": float(grader.metric_ROC(labels, score)),
        "PointF1_opt": float(grader.metric_PointF1(labels, score)),
    }
    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(labels, score, sliding_window)
    out["VUS_PR"] = float(vus_pr)
    out["VUS_ROC"] = float(vus_roc)
    return out


def eval_binary(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Point-level P/R/F1 from native binary predictions."""
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(preds).astype(bool)
    tp = int(np.sum(labels & preds))
    fp = int(np.sum(~labels & preds))
    fn = int(np.sum(labels & ~preds))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "F1": f1,
            "tp": tp, "fp": fp, "fn": fn}
