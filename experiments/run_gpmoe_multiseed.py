#!/usr/bin/env python3
"""GP-MoE (SMC, Zhang et al.) multi-seed variance on FIXED NEAL streams.

Replicates run_once from models/compare/GPMOE-main/.../pymp_GPMOE.py __main__
but separates the data seed (fixed stream, data_seed=409, same as the
GuardGP/RCGP fixed-stream experiments) from the model seed driving the SMC
inference. This quantifies the run-to-run variance of the stochastic baseline
on a fixed stream, complementing the finding that GuardGP is deterministic.

Writes results/gpmoe_multiseed.csv.
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from numpy.random import RandomState

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPMOE_DIR = os.path.join(REPO, "models", "compare", "GPMOE-main", "GPMOE-main", "code")
sys.path.insert(0, GPMOE_DIR)
sys.path.insert(0, REPO)

from pymp_GPMOE import ParticleGPMOE                       # noqa: E402
from dataset.neal.dataset_neal import (                    # noqa: E402
    make_stream_dataset_multioutlier_clean, neal_func)

DATA_SEED = 409
MODEL_SEEDS = list(range(10))
RATIOS = [0.1, 0.4]
OTYPE = "asymmetric"
EPS = 1e-12
OUT = os.path.join(REPO, "results", "gpmoe_multiseed.csv")


def run_once(model_seed, data_seed, otype, ratio):
    rng = RandomState(model_seed)
    torch.manual_seed(model_seed)

    X, y, _ = make_stream_dataset_multioutlier_clean(
        n_total=500, outlier_ratio=ratio, outlier_type=otype,
        normal_noise=0.2, x_range=(-3, 3),
        shuffle=True, shuffle_tail_only=True,
        clean_prefix=40, random_seed=data_seed,
        to_tensor=True, dtype=torch.float64, device="cpu",
    )
    X_np = X.cpu().numpy().reshape(-1, 1)
    Y_np = y.cpu().numpy().reshape(-1, 1)
    N, D = X_np.shape
    init_N = 40

    X_std = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-12)
    Y_std = (Y_np - float(Y_np.mean())) / (float(Y_np.std()) + 1e-12)
    y_mean_ref = float(Y_np[:init_N].mean())
    y_std_ref = float(Y_np[:init_N].std() + 1e-12)

    tpmoe = ParticleGPMOE(
        rng=rng, num_threads=1,
        X=X_std[0, None], Y=Y_std[0, None],
        J=10, alpha=1, X_mean=np.zeros(D), prior_obs=1,
        nu=3, psi=.5 * np.eye(D), alpha_a=10, alpha_b=1, mb_size=6,
    )

    f_warm = neal_func(X_np[:init_N].reshape(-1))
    mu_b = float(np.mean(f_warm))
    var_b = max(float(np.var(f_warm, ddof=1)), EPS)
    const_b = 0.5 * math.log(2.0 * math.pi * var_b)

    mse_sum = msll_sum = 0.0
    n_pred = 0
    step_ms = []
    for i in range(1, N):
        t0 = time.perf_counter()
        m, v = tpmoe.predict(X_std[i, None])
        mu_raw = float(m.ravel()[0]) * y_std_ref + y_mean_ref
        var_raw = max(float(v.ravel()[0]), EPS) * (y_std_ref ** 2)
        if i >= init_N:
            y_eval = float(neal_func(float(X_np[i, 0])))
            err = y_eval - mu_raw
            mse_sum += err * err
            nll_i = (0.5 * math.log(2.0 * math.pi * max(var_raw, EPS))
                     + 0.5 * err * err / max(var_raw, EPS))
            msll_sum += nll_i - (const_b + 0.5 * (y_eval - mu_b) ** 2 / var_b)
            n_pred += 1
        tpmoe.particle_update(X_std[i, None], Y_std[i, None])
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    return dict(
        method="gpmoe", dataset="neal", protocol="fixed-stream",
        outlier_type=otype, outlier_ratio=ratio,
        seed=model_seed, data_seed=data_seed,
        rmse=math.sqrt(mse_sum / max(n_pred, 1)),
        msll=msll_sum / max(n_pred, 1),
        avg_step_ms=float(np.mean(step_ms)),
    )


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for ratio in RATIOS:
        for sd in MODEL_SEEDS:
            r = run_once(sd, DATA_SEED, OTYPE, ratio)
            rows.append(r)
            print(f"ratio={ratio} seed={sd}: rmse={r['rmse']:.4f} "
                  f"msll={r['msll']:.3f} step={r['avg_step_ms']:.1f}ms "
                  f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\n=== variance across model seeds (fixed stream) ===")
    print(df.groupby("outlier_ratio")[["rmse", "msll"]]
            .agg(["mean", "std"]).round(4).to_string())
    print(f"\nsaved {OUT}")
