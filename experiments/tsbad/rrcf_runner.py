"""Streaming RRCF (Robust Random Cut Forest, Guha et al. 2016) baseline.

Single-pass protocol identical to GuardGP: the warm-up prefix is inserted
into the forest (scores recorded but not evaluated), then each new point is
scored by mean collusive displacement (CoDisp) and inserted with a sliding
tree budget. Uses the `rrcf` reference implementation.
"""

from collections import deque

import numpy as np
import rrcf


def run_rrcf_series(y: np.ndarray, warmup_n: int, *,
                    num_trees: int = 40,
                    tree_size: int = 256,
                    shingle_size: int = 8,
                    seed: int = 0) -> np.ndarray:
    """Return per-point anomaly scores (length = len(y)); the first
    (shingle_size - 1) points inherit the first computable score."""
    y = np.asarray(y, float)
    n = len(y)
    mu = float(np.mean(y[:warmup_n]))
    sd = float(np.std(y[:warmup_n]) + 1e-9)
    z = (y - mu) / sd

    rng = np.random.default_rng(seed)
    forest = [rrcf.RCTree(random_state=int(rng.integers(0, 2**31 - 1)))
              for _ in range(num_trees)]

    scores = np.zeros(n)
    first = shingle_size - 1
    inserted = deque()
    for t in range(first, n):
        point = z[t - first:t + 1]
        if len(inserted) >= tree_size:
            old = inserted.popleft()
            for tree in forest:
                tree.forget_point(old)
        codisp_sum = 0.0
        for tree in forest:
            tree.insert_point(point, index=t)
            codisp_sum += tree.codisp(t)
        scores[t] = codisp_sum / num_trees
        inserted.append(t)
    scores[:first] = scores[first]
    return scores
