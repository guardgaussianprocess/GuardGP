"""TSB-AD-U (NAB + Yahoo) evaluation harness.

Protocol:
  * Series: the published TSB-AD-U split, NAB and YAHOO families.
    Tuning list -> hyperparameter selection for RRCF and SPOT only;
    Eva list   -> reported results. GuardGP uses ONE default config
    everywhere (the paper's NAB batch defaults) and is never tuned.
  * Streaming: warm-up = min(train_index_from_filename, 500); all methods
    see the same warm-up prefix and are evaluated on t >= warmup only.
  * Metrics (TSB-AD code, vendored): VUS-PR (headline), VUS-ROC, AUC-PR,
    AUC-ROC, PointF1 at oracle threshold; plus native binary point-level
    P/R/F1 for GuardGP (alarm route) and SPOT (threshold exceedance).
    Sliding window via find_length_rank(rank=1) on the full series.

Usage (from this directory):
  python run_tsbad_eval.py --method guardgp --split both
  python run_tsbad_eval.py --method rrcf   --split tuning   # HP grid
  python run_tsbad_eval.py --method rrcf   --split eva --shingle 8 --tree-size 256 --seeds 0 1 2
  python run_tsbad_eval.py --method spot   --split tuning   # HP grid
  python run_tsbad_eval.py --method spot   --split eva --q 1e-4 --depth 200

Rows are appended to results/tsbad_raw.csv; existing (series, method,
config, seed) rows are skipped, so reruns are incremental.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(REPO, "dataset", "tsbad_u")
RESULTS = os.path.join(REPO, "results", "tsbad_raw.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from metrics_wrapper import eval_scores, eval_binary, sliding_window_of  # noqa: E402

WARMUP_CAP = 500


# ---------------------------------------------------------------------- #
# series enumeration
# ---------------------------------------------------------------------- #
def list_series(split: str):
    """split in {'tuning', 'eva', 'both'} -> list of dicts."""
    lists = {}
    for name in ("Eva", "Tuning"):
        with open(os.path.join(DATA_DIR, f"TSB-AD-U-{name}.csv")) as f:
            lists[name.lower()] = set(x.strip() for x in f if "_id_" in x)
    out = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_id_*.csv"))):
        base = os.path.basename(path)
        m = re.match(r"\d+_(\w+?)_id_.*_tr_(\d+)_1st_(\d+)\.csv", base)
        if not m:
            continue
        family, tr = m.group(1), int(m.group(2))
        member = "eva" if base in lists["eva"] else (
            "tuning" if base in lists["tuning"] else None)
        if member is None:
            continue
        if split != "both" and member != split:
            continue
        out.append(dict(path=path, series=base, family=family,
                        tr=tr, split=member))
    return out


def load_series(path):
    d = pd.read_csv(path)
    return d["Data"].to_numpy(float), d["Label"].to_numpy(int)


# ---------------------------------------------------------------------- #
# SPOT with a native score (threshold-exceedance ratio)
# ---------------------------------------------------------------------- #
def run_spot_series(y, warmup_n, *, q=1e-4, depth=200):
    from spot_baseline import BiDSPOT
    y = np.asarray(y, float)
    n = len(y)
    det = BiDSPOT(q=q, depth=depth)
    det.initialize(y[:warmup_n])
    scores = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)
    for t in range(warmup_n, n):
        m = float(np.mean(det.window))
        r = y[t] - m
        z_up = max(det.up.z, 1e-12)
        z_dn = max(det.dn.z, 1e-12)
        scores[t] = max(r / z_up, -r / z_dn)
        alarms[t] = det.step(y[t])
    return scores, alarms


# ---------------------------------------------------------------------- #
# harness
# ---------------------------------------------------------------------- #
def existing_keys():
    if not os.path.exists(RESULTS):
        return set()
    df = pd.read_csv(RESULTS)
    return set(zip(df["series"], df["method"], df["config"], df["seed"]))


def append_row(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(RESULTS)
    df.to_csv(RESULTS, mode="a", header=header, index=False)


def evaluate_one(entry, method, config_str, seed, scores, alarms, elapsed_s):
    y, labels = load_series(entry["path"])
    warm = min(entry["tr"], WARMUP_CAP)
    sw = sliding_window_of(y)
    lab_eval = labels[warm:]
    row = dict(series=entry["series"], family=entry["family"],
               split=entry["split"], method=method, config=config_str,
               seed=seed, warmup=warm, n=len(y), sliding_window=sw,
               n_anom_eval=int(lab_eval.sum()), elapsed_s=round(elapsed_s, 1))
    if lab_eval.sum() == 0:
        row.update(VUS_PR=np.nan, VUS_ROC=np.nan, AUC_PR=np.nan,
                   AUC_ROC=np.nan, PointF1_opt=np.nan)
    else:
        row.update(eval_scores(lab_eval, scores[warm:], sw))
    if alarms is not None:
        b = eval_binary(lab_eval, alarms[warm:])
        row.update(precision=b["precision"], recall=b["recall"], F1=b["F1"],
                   tp=b["tp"], fp=b["fp"], fn=b["fn"])
    append_row(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["guardgp", "rrcf", "spot", "sand", "olad"])
    ap.add_argument("--split", default="both",
                    choices=["tuning", "eva", "both"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    # rrcf
    ap.add_argument("--shingle", type=int, nargs="*", default=None)
    ap.add_argument("--tree-size", type=int, nargs="*", default=None)
    # spot
    ap.add_argument("--q", type=float, nargs="*", default=None)
    ap.add_argument("--depth", type=int, nargs="*", default=None)
    # olad
    ap.add_argument("--window", type=int, nargs="*", default=None)
    ap.add_argument("--sgd-lr", type=float, nargs="*", default=None)
    args = ap.parse_args()

    entries = list_series(args.split)
    done = existing_keys()
    print(f"[{args.method}] {len(entries)} series, split={args.split}")

    if args.method == "guardgp":
        from guardgp_runner import run_guardgp_series
        configs = [("default", {})]
    elif args.method == "rrcf":
        shingles = args.shingle or [4, 8, 16]
        tsizes = args.tree_size or [256]
        configs = [(f"sh{s}_ts{ts}", dict(shingle_size=s, tree_size=ts))
                   for s in shingles for ts in tsizes]
    elif args.method == "spot":
        qs = args.q or [1e-2, 1e-3, 1e-4]
        depths = args.depth or [50, 200]
        configs = [(f"q{q:g}_d{d}", dict(q=q, depth=d))
                   for q in qs for d in depths]
    elif args.method == "sand":
        configs = [("tsbad_default", {})]
    else:  # olad
        windows = args.window or [100, 200]
        lrs = args.sgd_lr if args.sgd_lr is not None else [0.0, 0.01]
        configs = [(f"w{w}_lr{lr:g}", dict(window=w, sgd_lr=lr))
                   for w in windows for lr in lrs]

    for cfg_str, cfg in configs:
        for seed in args.seeds:
            for k, e in enumerate(entries, 1):
                key = (e["series"], args.method, cfg_str, seed)
                if key in done:
                    continue
                y, _ = load_series(e["path"])
                warm = min(e["tr"], WARMUP_CAP)
                t0 = time.time()
                try:
                    if args.method == "guardgp":
                        res = run_guardgp_series(y, warm, seed=seed)
                        scores, alarms = res["scores"], res["alarms"]
                    elif args.method == "rrcf":
                        from rrcf_runner import run_rrcf_series
                        scores = run_rrcf_series(y, warm, seed=seed, **cfg)
                        alarms = None
                    elif args.method == "spot":
                        scores, alarms = run_spot_series(y, warm, **cfg)
                        if seed != args.seeds[0]:
                            continue  # deterministic
                    elif args.method == "sand":
                        from sand_runner import run_sand_series
                        scores = run_sand_series(y, warm, init_n=e["tr"],
                                                 seed=seed)
                        alarms = None
                    else:
                        from olad_runner import run_olad_series
                        scores, alarms = run_olad_series(y, warm, seed=seed, **cfg)
                        if seed != args.seeds[0]:
                            continue  # deterministic given stream
                    row = evaluate_one(e, args.method, cfg_str, seed,
                                       scores, alarms, time.time() - t0)
                    print(f"  [{k}/{len(entries)}] {cfg_str} s{seed} "
                          f"{e['series'][:40]:40s} VUS-PR={row.get('VUS_PR', float('nan')):.3f} "
                          f"({row['elapsed_s']}s)", flush=True)
                except Exception as ex:
                    print(f"  [{k}/{len(entries)}] {cfg_str} s{seed} "
                          f"{e['series'][:40]} FAILED: {ex}", flush=True)


if __name__ == "__main__":
    main()
