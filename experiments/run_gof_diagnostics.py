"""Goodness-of-fit diagnostics for the xi=0 (exponential) POT model.

Records the scores E_t actually computed by the GuardGP pipeline on clean
NEAL streams (outlier_ratio = 0, i.e. the nominal model holds) and tests:

(a) exactness claim: the raw PIT score E_raw = -log(2(1-Phi(z))) computed
    from the same (y, mu, vy) triples but WITHOUT the sigma floor and the
    1.3 inflation, against Exp(1) (fully specified null; bootstrap KS p).

(b) gate assumption: threshold excesses of the deployed (floored, inflated)
    scores over the gate's init quantile (0.90), against Exp(beta) with
    beta estimated by MLE (mean excess) -- Lilliefors-style parametric
    bootstrap KS p, plus coefficient of variation (=1 for exponential) and
    the exponential QQ-plot correlation coefficient.

Usage (from repo root):  python experiments/run_gof_diagnostics.py
Output: results/gof_diagnostics.csv
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.GuardGP as G

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
INIT_QUANTILE = 0.90
N_BOOT = 2000
RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------------- #
# score recorder
# ---------------------------------------------------------------------- #
RECORDS = []
_orig_surprise = G.surprise_from_gaussian


def _recording_surprise(y, mu, var, **kw):
    E = _orig_surprise(y, mu, var, **kw)
    RECORDS.append((float(y), float(mu), float(var), E))
    return E


def _raw_pit_score(y, mu, var):
    """E = -log(2(1-Phi(z))) with z = |y-mu|/sqrt(vy): no floor, no inflation."""
    return _orig_surprise(y, mu, var, sigma_floor=None, inflate=1.0)


# ---------------------------------------------------------------------- #
# KS helpers
# ---------------------------------------------------------------------- #
def _ks_stat_exp(x, scale):
    x = np.sort(np.asarray(x, float))
    n = x.size
    cdf = 1.0 - np.exp(-x / scale)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return max(d_plus, d_minus)


def ks_exp1_bootstrap(x, n_boot=N_BOOT):
    """KS test against Exp(1), fully specified null; bootstrap p-value."""
    n = len(x)
    d_obs = _ks_stat_exp(x, 1.0)
    d_boot = np.array([
        _ks_stat_exp(RNG.exponential(1.0, n), 1.0) for _ in range(n_boot)
    ])
    return d_obs, float(np.mean(d_boot >= d_obs))


def ks_exp_lilliefors(x, n_boot=N_BOOT):
    """KS test against Exp(scale) with scale estimated by MLE (mean);
    Lilliefors-style parametric bootstrap p-value."""
    n = len(x)
    d_obs = _ks_stat_exp(x, np.mean(x))
    d_boot = np.empty(n_boot)
    for b in range(n_boot):
        s = RNG.exponential(1.0, n)
        d_boot[b] = _ks_stat_exp(s, np.mean(s))
    return d_obs, float(np.mean(d_boot >= d_obs))


def qq_corr_exp(x):
    """Correlation coefficient of the exponential QQ-plot."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    q = -np.log(1.0 - (np.arange(1, n + 1) - 0.5) / n)
    return float(np.corrcoef(q, x)[0, 1])


# ---------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------- #
def main():
    from experiments.run_multiseed import GUARDGP_NEAL_CFG

    G.surprise_from_gaussian = _recording_surprise
    rows = []
    for data_seed in range(10):
        RECORDS.clear()
        res = G.run_guardgp(
            n_total=500, clean_prefix=40,
            outlier_ratio=0.0, outlier_type="asymmetric",
            seed=0, data_seed=data_seed,
            verbose=False, plot=False,
            **GUARDGP_NEAL_CFG,
        )
        rec = list(RECORDS)
        e_pipe = np.array([E for (_, _, _, E) in rec])
        e_raw = np.array([_raw_pit_score(y, mu, vy) for (y, mu, vy, _) in rec])

        # (a) raw PIT scores vs Exp(1)
        d_raw, p_raw = ks_exp1_bootstrap(e_raw)

        # (b) excesses of the deployed scores over the 0.90 init quantile
        u = float(np.quantile(e_pipe, INIT_QUANTILE))
        exc = e_pipe[e_pipe > u] - u
        beta = float(np.mean(exc))
        d_exc, p_exc = ks_exp_lilliefors(exc)
        cv = float(np.std(exc, ddof=1) / np.mean(exc))
        r_qq = qq_corr_exp(exc)

        rows.append(dict(
            data_seed=data_seed, n_scores=len(e_pipe),
            mean_E_raw=float(np.mean(e_raw)),
            ks_raw_exp1=d_raw, p_raw_exp1=p_raw,
            n_excess=len(exc), beta_hat=beta,
            ks_excess=d_exc, p_excess=p_exc,
            cv_excess=cv, qq_corr_excess=r_qq,
            rmse=res["rmse"], f1=res["f1"],
        ))
        print(f"[seed {data_seed}] n={len(e_pipe)}  mean(E_raw)={rows[-1]['mean_E_raw']:.3f}  "
              f"KS(raw,Exp1) p={p_raw:.3f}  |  n_exc={len(exc)}  beta={beta:.3f}  "
              f"KS(exc) p={p_exc:.3f}  CV={cv:.3f}  QQr={r_qq:.4f}")

    G.surprise_from_gaussian = _orig_surprise
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "gof_diagnostics.csv")
    df.to_csv(out, index=False)
    print("\nSummary over 10 clean streams:")
    print(df[["mean_E_raw", "p_raw_exp1", "beta_hat", "p_excess",
              "cv_excess", "qq_corr_excess"]].describe().loc[["mean", "min", "max"]].round(3).to_string())
    print(f"\nStreams not rejected at 5% -- raw vs Exp(1): "
          f"{int((df.p_raw_exp1 >= 0.05).sum())}/10, excesses exponential: "
          f"{int((df.p_excess >= 0.05).sum())}/10")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
