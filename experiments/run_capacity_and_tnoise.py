#!/usr/bin/env python3
"""Two remaining sensitivity studies for the rebuttal (Reviewer iAFA).

Part A -- capacity grid: expert capacity C (shard_cap_main; seed size scaled
as 40% of C as in the default 20/50) and EVT peak-buffer size P (max_peaks),
each varied with everything else at defaults, on NEAL 40% contamination,
3 outlier types x 10 random streams.

Part B -- misspecified nominal noise: NEAL with Student-t (df=3) nominal
noise of the same scale instead of Gaussian, 3 types x {10,40}% x 10 streams.

Writes results/capacity_tnoise.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import models.GuardGP as G
from models.evt_spot import SpotEVTExp
from experiments.run_multiseed import GUARDGP_NEAL_CFG

SEEDS = list(range(10))
OUT = os.path.join(REPO, "results", "capacity_tnoise.csv")
rows = []


def run_cfg(seed, otype, ratio, tag, **overrides):
    cfg = dict(GUARDGP_NEAL_CFG)
    cfg.update(overrides)
    res = G.run_guardgp(
        n_total=500, clean_prefix=40,
        outlier_ratio=ratio, outlier_type=otype,
        seed=seed, verbose=False, plot=False, **cfg,
    )
    return dict(part=tag, outlier_type=otype, outlier_ratio=ratio, seed=seed,
                rmse=res["rmse"], msll=res["msll"], f1=res["f1"])


def report(tag, key, val):
    m = pd.DataFrame([r for r in rows if r["part"] == tag])
    print(f"[{tag}] {key}={val}: rmse={m[m.part==tag].rmse.mean():.4f} "
          f"f1={m[m.part==tag].f1.mean():.3f} (cumulative)", flush=True)


# ---- Part A1: capacity C grid (seed size = 0.4 C, as 20/50 default) ----
for C in [25, 50, 100]:
    tag = f"C={C}"
    for otype in ["asymmetric", "focused", "uniform"]:
        for sd in SEEDS:
            rows.append(run_cfg(sd, otype, 0.4, tag,
                                shard_cap_main=C, seed_min=max(int(0.4 * C), 2)))
    m = pd.DataFrame([r for r in rows if r["part"] == tag])
    print(f"[capacity] C={C}: rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} "
          f"f1={m.f1.mean():.3f}", flush=True)

# ---- Part A2: peak-buffer P grid (monkeypatch SpotEVTExp max_peaks) ----
_orig_init = SpotEVTExp.__init__

def patched_init_factory(P):
    def __init__(self, q_hi=0.005, q_lo=0.02, init_quantile=0.90,
                 min_peaks=10, max_peaks=50):
        _orig_init(self, q_hi=q_hi, q_lo=q_lo, init_quantile=init_quantile,
                   min_peaks=min_peaks, max_peaks=P)
    return __init__

for P in [15, 30, 60, 120]:
    tag = f"P={P}"
    SpotEVTExp.__init__ = patched_init_factory(P)
    try:
        for otype in ["asymmetric", "focused", "uniform"]:
            for sd in SEEDS:
                rows.append(run_cfg(sd, otype, 0.4, tag))
    finally:
        SpotEVTExp.__init__ = _orig_init
    m = pd.DataFrame([r for r in rows if r["part"] == tag])
    print(f"[peakbuf] P={P}: rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} "
          f"f1={m.f1.mean():.3f}", flush=True)

# ---- Part B: Student-t nominal noise (df=3), same scale ----
_orig_maker = G.make_stream_dataset_multioutlier_clean

def t_noise_maker(**kw):
    """Same generator, but nominal Gaussian noise replaced by Student-t(3)
    of matching scale (sigma * sqrt((df-2)/df))."""
    noise = float(kw.get("normal_noise", 0.2))
    kw2 = dict(kw)
    kw2["normal_noise"] = 0.0
    X, y, is_out = _orig_maker(**kw2)
    rng = np.random.default_rng(int(kw.get("random_seed", 0)) + 555)
    df_t = 3.0
    scale = noise * np.sqrt((df_t - 2.0) / df_t)
    mask = ~is_out.numpy()
    t_noise = rng.standard_t(df_t, size=int(mask.sum())) * scale
    y[torch.tensor(mask)] += torch.tensor(t_noise, dtype=y.dtype)
    return X, y, is_out

G.make_stream_dataset_multioutlier_clean = t_noise_maker
try:
    for ratio in [0.1, 0.4]:
        tag = f"tnoise-{int(ratio*100)}"
        for otype in ["asymmetric", "focused", "uniform"]:
            for sd in SEEDS:
                rows.append(run_cfg(sd, otype, ratio, tag))
        m = pd.DataFrame([r for r in rows if r["part"] == tag])
        print(f"[t-noise] ratio={ratio:.0%}: rmse={m.rmse.mean():.4f}±{m.rmse.std():.4f} "
              f"f1={m.f1.mean():.3f}", flush=True)
finally:
    G.make_stream_dataset_multioutlier_clean = _orig_maker

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nsaved {OUT}")
