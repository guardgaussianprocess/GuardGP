#!/usr/bin/env python3
"""Warm-up sensitivity study for GuardGP on NEAL (Reviewer K2b4, C3/Q2).

Part A -- warm-up LENGTH: clean warm-up of {20, 40, 80} samples, stream
contamination 40%, three outlier types, 10 random streams each.

Part B -- warm-up CONTAMINATION: warm-up of 40 samples of which
{0, 10, 20, 30}% are corrupted with the same asymmetric outlier mechanism
(3-5 sigma shifts), stream contamination 40%, 10 random streams each.
Implemented by wrapping the dataset generator so that a fraction of the
warm-up prefix is corrupted after generation (evaluation is unchanged: the
warm-up segment is never scored).

Writes results/warmup_sensitivity.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import models.GuardGP as G
from experiments.run_multiseed import GUARDGP_NEAL_CFG

SEEDS = list(range(10))
RATIO = 0.4
OUT = os.path.join(REPO, "results", "warmup_sensitivity.csv")

_orig_maker = G.make_stream_dataset_multioutlier_clean


def make_contaminated_warmup(rho_w):
    """Return a dataset maker that corrupts a fraction rho_w of the warm-up."""
    def maker(**kw):
        X, y, is_out = _orig_maker(**kw)
        prefix = int(kw.get("clean_prefix", 0))
        n_bad = int(round(rho_w * prefix))
        if n_bad > 0:
            rng = np.random.default_rng(int(kw.get("random_seed", 0)) + 777)
            idx = rng.choice(prefix, n_bad, replace=False)
            std_y = float(y[:prefix].std().item())
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_bad)
            y[idx] = y[idx] + torch.tensor(shift, dtype=y.dtype)
            # warm-up points are never part of the scored stream; labels for
            # them are irrelevant to P/R/F1, so is_out is left unchanged.
        return X, y, is_out
    return maker


def run_one(seed, warmup, otype):
    res = G.run_guardgp(
        n_total=500, clean_prefix=warmup,
        outlier_ratio=RATIO, outlier_type=otype,
        seed=seed, verbose=False, plot=False,
        **GUARDGP_NEAL_CFG,
    )
    return dict(seed=seed, warmup=warmup, outlier_type=otype,
                rmse=res["rmse"], msll=res["msll"], f1=res["f1"],
                precision=res["precision"], recall=res["recall"])


rows = []

# ---- Part A: warm-up length ----
for warmup in [20, 40, 80]:
    for otype in ["asymmetric", "focused", "uniform"]:
        for sd in SEEDS:
            r = run_one(sd, warmup, otype)
            r.update(part="length", warmup_contam=0.0)
            rows.append(r)
        m = pd.DataFrame([x for x in rows if x["part"] == "length"
                          and x["warmup"] == warmup and x["outlier_type"] == otype])
        print(f"[length] warmup={warmup} {otype}: "
              f"rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} f1={m.f1.mean():.3f}",
              flush=True)

# ---- Part B: warm-up contamination ----
for rho_w in [0.0, 0.1, 0.2, 0.3]:
    G.make_stream_dataset_multioutlier_clean = make_contaminated_warmup(rho_w)
    try:
        for otype in ["asymmetric", "uniform"]:
            for sd in SEEDS:
                r = run_one(sd, 40, otype)
                r.update(part="contam", warmup_contam=rho_w)
                rows.append(r)
            m = pd.DataFrame([x for x in rows if x["part"] == "contam"
                              and x["warmup_contam"] == rho_w
                              and x["outlier_type"] == otype])
            print(f"[contam] rho_w={rho_w:.0%} {otype}: "
                  f"rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} f1={m.f1.mean():.3f}",
                  flush=True)
    finally:
        G.make_stream_dataset_multioutlier_clean = _orig_maker

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nsaved {OUT}")
