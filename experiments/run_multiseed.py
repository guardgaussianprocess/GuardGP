#!/usr/bin/env python3
"""Multi-seed experiment harness for the GuardGP rebuttal.

Runs a method over multiple *model* seeds on fixed data streams (the setup
requested by Reviewer Xr8x: "re-running each method with different
initialization and optimization seeds on the fixed streams"), appends raw
per-run rows to a CSV, and summarizes mean+-std with paired Wilcoxon
signed-rank tests between methods.

Usage examples (run from the repo root):

  # RCGP baseline on NEAL, full rebuttal sweep (3 types x 8 ratios x 10 seeds)
  python3 experiments/run_multiseed.py run --method rcgp --dataset neal \
      --seeds 0-9 --data-seed 409 \
      --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
      --types asymmetric,focused,uniform

  # GuardGP on the SAME fixed NEAL streams, 10 model seeds
  python3 experiments/run_multiseed.py run --method guardgp --dataset neal \
      --seeds 0-9 --data-seed 409 \
      --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
      --types asymmetric,focused,uniform

  # RCGP on all Franka joints, 10 seeds
  python3 experiments/run_multiseed.py run --method rcgp --dataset franka --seeds 0-9

  # Aggregate + significance + LaTeX rows
  python3 experiments/run_multiseed.py summarize

Notes
-----
* NAB/Yahoo GuardGP runners live in the full (non-public) codebase; register
  them in RUNNERS below with the same signature and the harness will pick
  them up unchanged.
* Raw rows are appended to results/raw_runs.csv, so runs can be resumed /
  extended; duplicate (method, dataset, config, seed) rows are dropped at
  summary time, keeping the last.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw_runs.csv")

CONFIG_COLS = ["method", "dataset", "protocol", "outlier_type", "outlier_ratio", "stream"]
METRICS = ["rmse", "msll", "f1", "precision", "recall", "avg_step_ms"]


# ---------------------------------------------------------------------- #
# method registry
# ---------------------------------------------------------------------- #
# Reproduction config from demo.ipynb (paper protocol on NEAL); the
# function defaults of run_guardgp do NOT reproduce the paper numbers.
GUARDGP_NEAL_CFG = dict(
    cover_frac=0.7, k_hist_main=0, alpha_main=0.5,
    k_hist_critic=0, alpha_critic=0.5,
    shard_cap_main=50, cap_critic=10, r_recent=100, seed_min=20,
    use_critic=True, use_critic_in_fusion=True,
    k_critic_seed_recent=10, critic_seed_min=10,
    critic_init_inherit_main=False, critic_rollover_inherit=True,
    opt_every_steps=1000, opt_epochs_main=0, opt_lr_main=0.05,
    opt_epochs_crit=0, opt_lr_crit=0.03,
    use_evt_currmain=True, use_evt_global=True,
)


def _run_guardgp_neal(seed, data_seed, ratio, otype, args):
    from models.GuardGP import run_guardgp
    res = run_guardgp(
        n_total=args.n_total, clean_prefix=args.warmup,
        outlier_ratio=ratio, outlier_type=otype,
        seed=seed, data_seed=data_seed,
        verbose=False, plot=False,
        **GUARDGP_NEAL_CFG,
    )
    res.update(dataset="neal", data_seed=data_seed, stream="")
    return res


def _run_rcgp_neal(seed, data_seed, ratio, otype, args):
    from experiments.runners import run_rcgp_neal
    res = run_rcgp_neal(
        n_total=args.n_total, clean_prefix=args.warmup,
        outlier_ratio=ratio, outlier_type=otype,
        seed=seed, data_seed=data_seed, capacity=args.capacity_neal,
    )
    res["stream"] = ""
    return res


def _run_rcgp_franka(seed, csv_path, args):
    from experiments.runners import run_rcgp_franka
    res = run_rcgp_franka(csv_path, seed=seed, capacity=args.capacity_franka)
    res.update(outlier_type="", outlier_ratio=float("nan"))
    return res


RUNNERS = {
    ("guardgp", "neal"): "neal",
    ("rcgp", "neal"): "neal",
    ("rcgp", "franka"): "franka",
    # Register full-codebase runners here, e.g.:
    # ("guardgp", "franka"): "franka",   -> add a _run_guardgp_franka above
}


# ---------------------------------------------------------------------- #
# statistics
# ---------------------------------------------------------------------- #
def wilcoxon_signed_rank(a, b):
    """Paired two-sided Wilcoxon signed-rank test (normal approximation with
    tie correction; zero differences dropped). Returns (W, p). For n < 6 the
    approximation is crude -- interpret with care."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return float("nan"), 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_pos = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(((counts ** 3 - counts).sum())) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var <= 0:
        return w_pos, 1.0
    z = (w_pos - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w_pos, p


# ---------------------------------------------------------------------- #
# run mode
# ---------------------------------------------------------------------- #
def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def cmd_run(args):
    key = (args.method, args.dataset)
    if key not in RUNNERS:
        sys.exit(f"No runner registered for method={args.method} dataset={args.dataset}. "
                 f"Available: {sorted(RUNNERS)}. For NAB/Yahoo/Franka GuardGP, register "
                 f"the full-codebase runner in RUNNERS (see header of this file).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    rows = []
    t_start = time.time()

    if args.dataset == "neal":
        ratios = [float(r) for r in args.ratios.split(",")]
        types = args.types.split(",")
        data_seed = None if args.vary_data else args.data_seed
        jobs = [(otype, ratio, sd) for otype in types for ratio in ratios for sd in seeds]
        for i, (otype, ratio, sd) in enumerate(jobs, 1):
            fn = _run_guardgp_neal if args.method == "guardgp" else _run_rcgp_neal
            res = fn(sd, data_seed, ratio, otype, args)
            res["method"] = args.method
            res["protocol"] = "vary-stream" if data_seed is None else "fixed-stream"
            rows.append(res)
            print(f"[{i}/{len(jobs)}] {args.method} neal {otype} {ratio:.0%} seed={sd} "
                  f"rmse={res['rmse']:.4f} msll={res['msll']:.3f} "
                  f"({time.time()-t_start:.0f}s elapsed)", flush=True)

    elif args.dataset == "franka":
        franka_dir = os.path.join(os.path.dirname(RESULTS_DIR), "dataset", "franka")
        ids = [int(x) for x in args.franka_ids.split(",")]
        files = [os.path.join(franka_dir, f"franka{k}_labeled.csv") for k in ids]
        jobs = [(f, sd) for f in files for sd in seeds]
        for i, (f, sd) in enumerate(jobs, 1):
            res = _run_rcgp_franka(sd, f, args)
            res["method"] = args.method
            res["protocol"] = "fixed-stream"
            rows.append(res)
            print(f"[{i}/{len(jobs)}] {args.method} {res['stream']} seed={sd} "
                  f"rmse={res['rmse']:.4f} msll={res['msll']:.3f} "
                  f"({time.time()-t_start:.0f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    drop = [c for c in ("detected_indices", "true_attack_indices") if c in df.columns]
    df = df.drop(columns=drop)
    if os.path.exists(RAW_CSV):
        existing_cols = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
        all_cols = existing_cols + [c for c in df.columns if c not in existing_cols]
        if all_cols != existing_cols:  # widen the file once so columns stay aligned
            full = pd.read_csv(RAW_CSV).reindex(columns=all_cols)
            full.to_csv(RAW_CSV, index=False)
        df = df.reindex(columns=all_cols)
        df.to_csv(RAW_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_CSV, index=False)
    print(f"\nAppended {len(df)} rows to {os.path.abspath(RAW_CSV)}")


# ---------------------------------------------------------------------- #
# summarize mode
# ---------------------------------------------------------------------- #
def cmd_summarize(args):
    if not os.path.exists(RAW_CSV):
        sys.exit(f"{RAW_CSV} not found -- run experiments first.")
    df = pd.read_csv(RAW_CSV)
    for c in CONFIG_COLS:
        if c not in df.columns:
            df[c] = ""
    df[CONFIG_COLS[2:]] = df[CONFIG_COLS[2:]].fillna("")
    df["protocol"] = df["protocol"].replace("", "fixed-stream")
    # keep last duplicate of (config, seed)
    df = df.drop_duplicates(subset=CONFIG_COLS + ["seed"], keep="last")

    metrics = [m for m in METRICS if m in df.columns]
    agg = df.groupby(CONFIG_COLS, dropna=False)[metrics].agg(["mean", "std", "count"])
    out_csv = os.path.join(RESULTS_DIR, "summary.csv")
    agg.to_csv(out_csv)
    print(f"Summary written to {os.path.abspath(out_csv)}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(agg.round(4))

    # paired significance between methods sharing the same (dataset, config, seed)
    print("\n=== Paired Wilcoxon signed-rank (per config, across seeds) ===")
    cfg_cols = CONFIG_COLS[1:]  # dataset + protocol + config, without method
    for cfg, sub in df.groupby(cfg_cols, dropna=False):
        methods = sorted(sub["method"].unique())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a = sub[sub["method"] == methods[i]].set_index("seed")
                b = sub[sub["method"] == methods[j]].set_index("seed")
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                for m in ("rmse", "msll", "f1"):
                    if m not in sub.columns:
                        continue
                    x, y = a.loc[common, m], b.loc[common, m]
                    if x.isna().all() or y.isna().all():
                        continue
                    _, p = wilcoxon_signed_rank(x.to_numpy(), y.to_numpy())
                    tag = "*" if p < 0.05 else " "
                    print(f"{cfg} | {m}: {methods[i]} {x.mean():.4f} vs "
                          f"{methods[j]} {y.mean():.4f}  p={p:.4f}{tag} (n={len(common)})")

    # LaTeX rows: mean +- std per (method, config)
    print("\n=== LaTeX rows (mean $\\pm$ std) ===")
    for (method, dataset, protocol, otype, ratio, stream), sub in df.groupby(CONFIG_COLS, dropna=False):
        cells = []
        for m in ("rmse", "msll", "f1"):
            if m in sub.columns and not sub[m].isna().all():
                cells.append(f"${sub[m].mean():.3f} \\pm {sub[m].std():.3f}$")
            else:
                cells.append("---")
        label = f"{method} [{protocol}] & {dataset} {otype} {ratio} {stream}".strip()
        print(f"{label} & " + " & ".join(cells) + r" \\")


# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    pr = sp.add_parser("run", help="run experiments and append to raw CSV")
    pr.add_argument("--method", required=True, choices=["guardgp", "rcgp"])
    pr.add_argument("--dataset", required=True, choices=["neal", "franka"])
    pr.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    pr.add_argument("--data-seed", type=int, default=409,
                    help="fixes the NEAL stream; model seeds vary independently")
    pr.add_argument("--vary-data", action="store_true",
                    help="vary the stream with the seed (paper protocol) instead of fixing it")
    pr.add_argument("--ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    pr.add_argument("--types", default="asymmetric,focused,uniform")
    pr.add_argument("--n-total", type=int, default=500)
    pr.add_argument("--warmup", type=int, default=40)
    pr.add_argument("--capacity-neal", type=int, default=60,
                    help="RCGP buffer capacity on NEAL (dataset budget: 60)")
    pr.add_argument("--capacity-franka", type=int, default=120,
                    help="RCGP buffer capacity on Franka (dataset budget: 120)")
    pr.add_argument("--franka-ids", default="1,2,3,4,5,6,7")
    pr.set_defaults(func=cmd_run)

    ps = sp.add_parser("summarize", help="aggregate raw CSV, significance, LaTeX")
    ps.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
