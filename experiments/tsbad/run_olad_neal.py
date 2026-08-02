"""OLAD (IJCAI'17 Student-t process, tuned config: window=100, OLAD-1) on
the NEAL contaminated-stream regression benchmark — identical streams,
warm-up, and metrics as the GuardGP/RCGP runs in results/raw_runs.csv
(protocol 'vary-stream': data_seed 0..9 varies the stream, model seed 0).

Appends rows with method='olad' to results/raw_runs.csv.
Run from experiments/tsbad/:  python run_olad_neal.py
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from olad_runner import run_olad_regression  # noqa: E402
from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func  # noqa: E402

RAW = os.path.join(REPO, "results", "raw_runs.csv")
EPS = 1e-12

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TYPES = ["asymmetric", "focused", "uniform"]
DATA_SEEDS = range(10)
CLEAN_PREFIX = 40
N_TOTAL = 500


def run_one(otype, ratio, data_seed):
    X, y, _ = make_stream_dataset_multioutlier_clean(
        n_total=N_TOTAL, outlier_ratio=ratio, outlier_type=otype,
        normal_noise=0.2, x_range=(-3, 3), shuffle=True,
        shuffle_tail_only=True, clean_prefix=CLEAN_PREFIX,
        random_seed=data_seed, to_tensor=True,
        dtype=torch.float64, device="cpu")
    X_np = X.cpu().numpy().squeeze()
    y_np = y.cpu().numpy().squeeze()
    y_true = neal_func(X_np)

    t0 = time.perf_counter()
    mu, sd, _ = run_olad_regression(X_np, y_np, CLEAN_PREFIX,
                                    window=100, sgd_lr=0.0, seed=0)
    dt = time.perf_counter() - t0

    yt_warm = y_true[:CLEAN_PREFIX]
    mu_b = float(np.mean(yt_warm))
    var_b = max(float(np.var(yt_warm, ddof=1)), EPS)
    const_b = 0.5 * math.log(2.0 * math.pi * var_b)

    mse = nll = msll = 0.0
    n_pred = 0
    for t in range(CLEAN_PREFIX, N_TOTAL):
        err = y_true[t] - mu[t]
        v = max(sd[t] ** 2, EPS)
        mse += err * err
        nll_i = 0.5 * math.log(2 * math.pi * v) + 0.5 * err * err / v
        nll += nll_i
        msll += nll_i - (const_b + 0.5 * (y_true[t] - mu_b) ** 2 / var_b)
        n_pred += 1
    mse /= n_pred
    return dict(
        rmse=math.sqrt(mse), smse=mse / var_b, msll=msll / n_pred,
        nll=nll / n_pred,
        f1=float("nan"), precision=float("nan"), recall=float("nan"),
        avg_step_ms=1000.0 * dt / n_pred, avg_pred_ms=float("nan"),
        avg_detect_ms=float("nan"), avg_upd_ms=float("nan"),
        method="olad", seed=0, outlier_type=otype, outlier_ratio=ratio,
        n_total=N_TOTAL, clean_prefix=CLEAN_PREFIX,
        route_main=float("nan"), route_critic=float("nan"),
        missed_attack_indices="", dataset="neal", data_seed=data_seed,
        stream="", protocol="vary-stream", obs="")


def main():
    done = set()
    if os.path.exists(RAW):
        df = pd.read_csv(RAW)
        d = df[df.method == "olad"]
        done = set(zip(d.outlier_type, d.outlier_ratio, d.data_seed))
    cols = pd.read_csv(RAW, nrows=0).columns.tolist()
    for otype in TYPES:
        for ratio in RATIOS:
            for ds in DATA_SEEDS:
                if (otype, ratio, ds) in done:
                    continue
                row = run_one(otype, ratio, ds)
                pd.DataFrame([row])[cols].to_csv(RAW, mode="a",
                                                 header=False, index=False)
            print(f"{otype} {ratio:.1f} done", flush=True)


if __name__ == "__main__":
    main()
