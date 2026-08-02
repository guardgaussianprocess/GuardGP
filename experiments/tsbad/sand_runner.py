"""SAND baseline (Boniol et al., VLDB 2021) via the TSB-AD implementation.

Invocation follows TSB-AD's model_wrapper.run_SAND exactly (online mode),
with init_length = the published train index of the series (TSB-AD's own
semi-supervised protocol; SAND requires an initialization batch well above
4x the pattern window, so the capped 500-point warm-up of the streaming
methods is not viable for it). Evaluation region stays identical to the
other methods. Score-only method (no native binary decisions).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsb_sand import SAND  # noqa: E402
from tsb_slidingwindows import find_length_rank  # noqa: E402


def run_sand_series(y: np.ndarray, warmup_n: int, *, init_n: int = None,
                    seed: int = 0) -> np.ndarray:
    y = np.asarray(y, float).squeeze()
    np.random.seed(seed)
    sw = find_length_rank(y.reshape(-1, 1), rank=1)
    clf = SAND(pattern_length=sw, subsequence_length=4 * sw)
    clf.fit(y, online=True,
            overlaping_rate=int(1.5 * sw),
            init_length=init_n or warmup_n,
            alpha=0.5,
            batch_size=max(5 * sw, int(0.1 * len(y))))
    score = np.asarray(clf.decision_scores_, float).ravel()
    if len(score) < len(y):  # pad tail/head convention differences
        pad = np.full(len(y) - len(score), score[-1])
        score = np.concatenate([score, pad])
    return score[:len(y)]
