"""Stream runners returning per-run metric dicts (same schema as run_guardgp)."""

from typing import Any, Dict, Optional

import math
import time

import numpy as np
import torch

from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func
from models.RCGP_Online import OnlineRCGP

EPS = 1e-12
DTYPE = torch.float64


def _stream_metrics_init(y_true_warm: np.ndarray):
    mu_b = float(np.mean(y_true_warm))
    var_b = max(float(np.var(y_true_warm, ddof=1)), EPS)
    const_b = 0.5 * math.log(2.0 * math.pi * var_b)
    return mu_b, var_b, const_b


def _rand_init(seed: int):
    """Per-seed random hyperparameter initialisation (the stream stays fixed;
    only initialisation/optimisation randomness varies across seeds)."""
    g = torch.Generator().manual_seed(int(seed))
    ls = float(torch.empty(1).uniform_(0.5, 2.0, generator=g).item())
    var = float(torch.empty(1).uniform_(0.5, 2.0, generator=g).item())
    noise = float(torch.empty(1).uniform_(0.1, 0.4, generator=g).item())
    return ls, var, noise


def run_rcgp_stream(
    X: torch.Tensor,
    y_obs: torch.Tensor,
    y_true: np.ndarray,
    warmup_n: int,
    *,
    capacity: int = 60,
    seed: int = 0,
    center: str = "pred",
    c_mode: str = "quantile",
    c_quantile: float = 0.95,
    c_mult: float = 2.0,
    opt_epochs: int = 200,
    opt_lr: float = 0.05,
) -> Dict[str, Any]:
    """Single-pass online RCGP over a fixed stream.

    X: (N, d) inputs; y_obs: (N,) possibly contaminated observations;
    y_true: (N,) clean ground truth used for offline evaluation only;
    warmup_n: length of the trusted warm-up prefix (identical protocol to
    all baselines: hyperparameters optimised once on this prefix).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = X.to(DTYPE)
    y_obs = y_obs.to(DTYPE)
    N, d = X.shape

    ls0, var0, noise0 = _rand_init(seed)
    model = OnlineRCGP(
        input_dim=d, capacity=capacity, center=center,
        c_mode=c_mode, c_quantile=c_quantile, c_mult=c_mult,
    )
    model.warmup_fit(
        X[:warmup_n], y_obs[:warmup_n],
        epochs=opt_epochs, lr=opt_lr,
        init_lengthscale=ls0, init_variance=var0, init_noise=noise0,
    )

    mu_b, var_b, const_b = _stream_metrics_init(np.asarray(y_true[:warmup_n], dtype=float))

    mse_sum = nll_sum = msll_sum = 0.0
    n_pred = 0
    t_steps = []

    for t in range(warmup_n, N):
        t0 = time.perf_counter()
        mu, _, vy = model.predict(X[t])
        model.update(X[t], float(y_obs[t].item()), mu_pred=mu)
        t_steps.append(time.perf_counter() - t0)

        yt = float(y_true[t])
        err = yt - mu
        v_eff = max(vy, EPS)
        mse_sum += err * err
        nll_i = 0.5 * math.log(2.0 * math.pi * v_eff) + 0.5 * (err * err) / v_eff
        nll_sum += nll_i
        nll_base_i = const_b + 0.5 * ((yt - mu_b) ** 2) / var_b
        msll_sum += nll_i - nll_base_i
        n_pred += 1

    avg_mse = mse_sum / max(n_pred, 1)
    return dict(
        method="RCGP",
        seed=seed,
        rmse=math.sqrt(max(avg_mse, 0.0)),
        smse=avg_mse / var_b,
        msll=msll_sum / max(n_pred, 1),
        nll=nll_sum / max(n_pred, 1),
        precision=float("nan"), recall=float("nan"), f1=float("nan"),
        avg_step_ms=1000.0 * float(np.mean(t_steps)) if t_steps else float("nan"),
        n_total=N,
        clean_prefix=warmup_n,
    )


# ---------------------------------------------------------------------- #
# NEAL
# ---------------------------------------------------------------------- #
def run_rcgp_neal(
    *,
    n_total: int = 500,
    clean_prefix: int = 50,
    outlier_ratio: float = 0.4,
    outlier_type: str = "asymmetric",
    seed: int = 0,
    data_seed: Optional[int] = None,
    normal_noise: float = 0.2,
    x_range=(-3, 3),
    capacity: int = 60,
    **rcgp_kwargs,
) -> Dict[str, Any]:
    """RCGP on the NEAL synthetic stream, same generator/protocol as run_guardgp.
    ``data_seed`` fixes the stream while ``seed`` varies the model
    initialisation; if None, data_seed = seed (matching run_guardgp)."""
    ds = seed if data_seed is None else data_seed
    X, y, is_out = make_stream_dataset_multioutlier_clean(
        n_total=n_total,
        outlier_ratio=outlier_ratio,
        outlier_type=outlier_type,
        normal_noise=normal_noise,
        x_range=x_range,
        shuffle=True,
        shuffle_tail_only=True,
        clean_prefix=clean_prefix,
        random_seed=ds,
        to_tensor=True,
        dtype=DTYPE,
        device="cpu",
    )
    y_true = neal_func(X.cpu().numpy().squeeze())
    res = run_rcgp_stream(X, y, y_true, clean_prefix, capacity=capacity,
                          seed=seed, **rcgp_kwargs)
    res.update(outlier_type=outlier_type, outlier_ratio=outlier_ratio,
               dataset="neal", data_seed=ds)
    return res


# ---------------------------------------------------------------------- #
# Franka
# ---------------------------------------------------------------------- #
FRANKA_WARMUP = 1100  # first 1100 samples are filtered signals (Appendix C)


def load_franka(csv_path: str, obs: str = "tau"):
    """Return (X 14-dim, y_obs, y_true filtered residual, labels).

    obs='tau' uses the raw torque tau_k as observation (matching the paper's
    GuardGP Franka pipeline, new_franka.py); obs='raw' uses tau_residual_raw_k.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns
                 if c.startswith("joint_pos_") or c.startswith("joint_vel_")]
    assert len(feat_cols) == 14, f"expected 14 joint features, got {len(feat_cols)}"
    tau_true_col = [c for c in df.columns
                    if c.startswith("tau_residual_") and "raw" not in c][0]
    if obs == "tau":
        tau_raw_col = [c for c in df.columns
                       if c.startswith("tau_") and "residual" not in c][0]
    else:
        tau_raw_col = [c for c in df.columns if c.startswith("tau_residual_raw_")][0]
    label_col = [c for c in df.columns if c.startswith("anomaly_")][0]

    X = torch.tensor(df[feat_cols].to_numpy(dtype=float), dtype=DTYPE)
    y_obs = torch.tensor(df[tau_raw_col].to_numpy(dtype=float), dtype=DTYPE)
    y_true = df[tau_true_col].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=int)
    return X, y_obs, y_true, labels


def run_rcgp_franka(
    csv_path: str,
    *,
    seed: int = 0,
    capacity: int = 120,
    warmup_n: int = FRANKA_WARMUP,
    standardize: bool = True,
    obs: str = "tau",
    **rcgp_kwargs,
) -> Dict[str, Any]:
    X, y_obs, y_true, _ = load_franka(csv_path, obs=obs)
    if standardize:  # warm-up statistics only (no lookahead)
        mu = X[:warmup_n].mean(dim=0, keepdim=True)
        sd = X[:warmup_n].std(dim=0, keepdim=True).clamp_min(1e-8)
        X = (X - mu) / sd
    res = run_rcgp_stream(X, y_obs, y_true, warmup_n, capacity=capacity,
                          seed=seed, **rcgp_kwargs)
    res.update(dataset="franka", stream=csv_path.split("/")[-1], obs=obs)
    return res
