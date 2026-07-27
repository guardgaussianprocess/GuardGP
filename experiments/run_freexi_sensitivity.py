#!/usr/bin/env python3
"""Free-xi GPD sensitivity for GuardGP's EVT gates (Xr8x C4, K2b4 C1/Q1).

Replaces the fixed exponential tail (xi = 0) in both gates by a free-xi GPD
fitted online with Grimshaw's method (same estimator as SPOT), leaving
everything else identical, and re-runs GuardGP on NEAL over the same 10
random streams per configuration as the main experiments. If the results
match the xi=0 runs, the fixed-xi choice is not load-bearing.

Writes results/freexi_sensitivity.csv.
"""

import math
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

import models.GuardGP as G
from models.evt_spot import SpotEVTExp
from spot_baseline import grimshaw
from experiments.run_multiseed import GUARDGP_NEAL_CFG

SEEDS = list(range(10))
RATIOS = [0.1, 0.4, 0.8]
TYPES = ["asymmetric", "focused", "uniform"]
OUT = os.path.join(REPO, "results", "freexi_sensitivity.csv")


class SpotEVTGrimshaw(SpotEVTExp):
    """SpotEVTExp with thresholds from a free-xi GPD (Grimshaw MLE) instead
    of the fixed exponential (xi=0) tail. Falls back to the exponential
    formula when the peak set is too small or the fit is degenerate."""

    def _update_thresholds(self):
        Nt = len(self.peaks)
        if Nt <= 0 or self.n_seen <= 0:
            self.z_lo = self.z_hi = float("inf")
            return
        peaks = np.asarray(self.peaks, float)
        if Nt >= 8:
            gamma, sigma = grimshaw(peaks)
        else:
            gamma, sigma = 0.0, max(peaks.mean(), 1e-9)

        def z_of(q):
            r = q * self.n_seen / Nt
            r = max(r, 1e-300)
            if abs(gamma) < 1e-12:
                return self.tE + sigma * max(math.log(1.0 / r), 0.0)
            return self.tE + (sigma / gamma) * max(r ** (-gamma) - 1.0, 0.0)

        self.z_lo = z_of(self.q_lo)
        self.z_hi = z_of(self.q_hi)


if __name__ == "__main__":
    rows = []
    G.SpotEVTExp = SpotEVTGrimshaw  # both gates use the free-xi fit
    try:
        for otype in TYPES:
            for ratio in RATIOS:
                for sd in SEEDS:
                    res = G.run_guardgp(
                        n_total=500, clean_prefix=40,
                        outlier_ratio=ratio, outlier_type=otype,
                        seed=sd, verbose=False, plot=False,
                        **GUARDGP_NEAL_CFG,
                    )
                    rows.append(dict(
                        xi="free", outlier_type=otype, outlier_ratio=ratio,
                        seed=sd, rmse=res["rmse"], msll=res["msll"],
                        f1=res["f1"], precision=res["precision"],
                        recall=res["recall"],
                    ))
                m = pd.DataFrame(rows)
                m = m[(m.outlier_type == otype) & (m.outlier_ratio == ratio)]
                print(f"[free-xi] {otype} {ratio:.0%}: "
                      f"rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} "
                      f"f1={m.f1.mean():.3f}", flush=True)
    finally:
        G.SpotEVTExp = SpotEVTExp

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # paired comparison against the xi=0 runs already in raw_runs.csv
    raw = pd.read_csv(os.path.join(REPO, "results", "raw_runs.csv"))
    base = raw[(raw.method == "guardgp") & (raw.dataset == "neal")
               & (raw.protocol == "vary-stream")
               & (raw.outlier_ratio.isin(RATIOS))]
    print("\n=== xi=0 (main runs) vs free-xi (Grimshaw), mean over 10 streams ===")
    for otype in TYPES:
        for ratio in RATIOS:
            b = base[(base.outlier_type == otype) & (base.outlier_ratio == ratio)]
            f = df[(df.outlier_type == otype) & (df.outlier_ratio == ratio)]
            if len(b) and len(f):
                print(f"{otype:11s} {ratio:.0%}: "
                      f"rmse {b.rmse.mean():.4f} -> {f.rmse.mean():.4f} | "
                      f"f1 {b.f1.mean():.3f} -> {f.f1.mean():.3f}")
    print(f"\nsaved {OUT}")
