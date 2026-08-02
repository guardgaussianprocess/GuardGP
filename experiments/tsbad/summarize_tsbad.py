"""Summarize results/tsbad_raw.csv.

1. Hyperparameter selection: for RRCF and SPOT, pick the config with the
   best mean VUS-PR (RRCF) / best mean point-F1 (SPOT, its native operating
   mode) on the TUNING split.
2. Report: Eva-split per-family (NAB / YAHOO) means of VUS-PR, VUS-ROC,
   AUC-PR, PointF1_opt, and native binary P/R/F1 where available, for
   GuardGP (default config) and the tuned baselines. RRCF metrics averaged
   over seeds per series first.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(REPO, "results", "tsbad_raw.csv")

SCORE_COLS = ["VUS_PR", "VUS_ROC", "AUC_PR", "AUC_ROC", "PointF1_opt"]
BIN_COLS = ["precision", "recall", "F1"]


def main():
    df = pd.read_csv(RAW)
    df = df.dropna(subset=["VUS_PR"])

    # -------- HP selection on tuning split --------
    picks = {"guardgp": "default", "sand": "tsbad_default"}
    tun = df[df.split == "tuning"]
    for method, crit in (("rrcf", "VUS_PR"), ("spot", "F1"), ("olad", "F1")):
        t = tun[tun.method == method]
        if t.empty:
            continue
        per_cfg = (t.groupby(["config", "series"])[crit].mean()
                     .groupby("config").mean()
                     .sort_values(ascending=False, kind="stable"))
        print(f"\n[{method}] tuning-split mean {crit} by config:")
        print(per_cfg.round(3).to_string())
        # ties (within 1e-6) broken toward a config that has eva rows
        eva_cfgs = set(df[(df.split == "eva") & (df.method == method)].config)
        best_val = per_cfg.iloc[0]
        tied = [c for c in per_cfg.index if per_cfg[c] >= best_val - 1e-6]
        in_eva = [c for c in tied if c in eva_cfgs]
        picks[method] = in_eva[0] if in_eva else per_cfg.index[0]
    print("\nselected configs:", picks)

    # -------- Eva-split report --------
    eva = df[df.split == "eva"]
    rows = []
    for method, cfg in picks.items():
        e = eva[(eva.method == method) & (eva.config == cfg)]
        if e.empty:
            continue
        # average over seeds per series first
        per_series = e.groupby(["family", "series"]).mean(numeric_only=True)
        for fam, gg in per_series.groupby(level="family"):
            r = dict(method=method, config=cfg, family=fam, n_series=len(gg))
            for c in SCORE_COLS + BIN_COLS:
                if c in gg and gg[c].notna().any():
                    r[c] = gg[c].mean()
            rows.append(r)
        r = dict(method=method, config=cfg, family="ALL",
                 n_series=len(per_series))
        for c in SCORE_COLS + BIN_COLS:
            if c in per_series and per_series[c].notna().any():
                r[c] = per_series[c].mean()
        rows.append(r)

    out = pd.DataFrame(rows)
    cols = ["method", "config", "family", "n_series"] + \
           [c for c in SCORE_COLS + BIN_COLS if c in out.columns]
    out = out[cols].round(3)
    print("\n===== Eva split (means over series) =====")
    print(out.to_string(index=False))
    out_path = os.path.join(REPO, "results", "tsbad_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
