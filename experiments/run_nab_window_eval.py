#!/usr/bin/env python3
"""Evaluate GuardGP and SPOT on NAB under BOTH protocols:

  * point protocol (paper): point-level precision, event-level recall
  * window protocol (standard NAB-style event evaluation): detections merged
    into events (gap <= 10 steps); TP = true windows hit by >= 1 detection,
    FP = detection events with no overlap with any true window, FN = missed
    windows; precision/recall/F1 at the event level.

Runs the paper's GuardGP NAB pipeline (models/new_NAB.py, batch defaults +
cover_frac=0.7 as in its __main__) to obtain detection indices, and BiDSPOT
(q=1e-4) from experiments/spot_baseline.py.

Writes results/nab_window_eval.csv.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")
sys.path.insert(0, MODELS)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from spot_baseline import BiDSPOT, mask_to_ranges, prf_point_event, NAB_STREAMS


def merge_by_steps(indices, max_gap=10):
    """Merge sorted detection indices into events [(s, e), ...]."""
    if not indices:
        return []
    idx = sorted(indices)
    events, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= max_gap:
            p = i
        else:
            events.append((s, p))
            s = p = i
    events.append((s, p))
    return events


def prf_window(det_idx, is_out, start, max_gap=10):
    """Event-level (window) protocol."""
    det_idx = [i for i in det_idx if i >= start]
    stream_mask = is_out.copy()
    stream_mask[:start] = False
    windows = mask_to_ranges(stream_mask)
    det_events = merge_by_steps(det_idx, max_gap=max_gap)

    def overlaps(ev, win):
        return ev[0] <= win[1] and win[0] <= ev[1]

    tp_windows = sum(1 for w in windows if any(overlaps(e, w) for e in det_events))
    fp_events = sum(1 for e in det_events if not any(overlaps(e, w) for w in windows))
    tp_events = len(det_events) - fp_events
    prec = tp_events / len(det_events) if det_events else 0.0
    rec = tp_windows / len(windows) if windows else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, len(det_events), fp_events, len(windows), tp_windows


def run_guardgp_nab():
    """Paper pipeline; returns list of (stream, det_idx, is_out, start)."""
    from new_NAB import run_gpevt_on_nab_shard_critic
    out = []
    cwd = os.getcwd()
    os.chdir(MODELS)
    try:
        for sp in NAB_STREAMS:
            res = run_gpevt_on_nab_shard_critic(
                nab_root="NAB", subpath=sp,
                labels_file="labels/combined_windows.json",
                warmup_hours=24.0, fallback_warmup_points=50,
                # batch-runner defaults (paper config)
                shard_cap_main=100, cap_critic=20,
                r_recent=100, k_recent=0, seed_min=60,
                k_hist_main=3, alpha_main=0.2,
                k_hist_critic=3, alpha_critic=0.2,
                k_critic_seed_recent=20, critic_seed_min=20,
                critic_init_inherit_main=False, critic_rollover_inherit=True,
                use_proto_neal=False, num_proto_neal=0,
                cover_frac=0.7,
                use_evt_currmain=True, use_evt_global=True,
                opt_every=12000, opt_epochs_main=0, opt_lr_main=0.05,
                opt_epochs_crit=0, opt_lr_crit=0.03,
                seed=0, normalize_y=True,
                verbose=False, plot=False,
                merge_by="steps", max_gap_steps=10, max_gap_hours=0.5,
                export_csv=False,
            )
            # rebuild the label mask exactly as the pipeline saw it
            from dataset_nab import make_stream_dataset_nab_windows
            _, _, _, is_out = make_stream_dataset_nab_windows(
                nab_root="NAB", subpath=sp,
                labels_file="labels/combined_windows.json",
                normalize_y=True, to_tensor=False,
            )
            is_out = np.asarray(is_out, bool).ravel()
            out.append((sp, list(res["detected_indices"]),
                        is_out, int(res["stream_start_idx"])))
            print(f"[GuardGP] {sp}: {len(res['detected_indices'])} detections, "
                  f"start={res['stream_start_idx']}", flush=True)
    finally:
        os.chdir(cwd)
    return out


def run_spot_nab(q=1e-4):
    from dataset_nab import make_stream_dataset_nab_windows
    out = []
    cwd = os.getcwd()
    os.chdir(MODELS)
    try:
        for sp in NAB_STREAMS:
            X_plot, _, y, is_out = make_stream_dataset_nab_windows(
                nab_root="NAB", subpath=sp,
                labels_file="labels/combined_windows.json",
                normalize_y=True, to_tensor=False,
            )
            y = np.asarray(y, float).ravel()
            is_out = np.asarray(is_out, bool).ravel()
            ts_h = np.asarray(X_plot, float).ravel()
            start = int(np.searchsorted(ts_h, ts_h[0] + 24.0))
            det = BiDSPOT(q=q, depth=min(max(start, 10), 300))
            det.initialize(y[:start])
            alarms = [t for t in range(start, len(y)) if det.step(y[t])]
            out.append((sp, alarms, is_out, start))
    finally:
        os.chdir(cwd)
    return out


if __name__ == "__main__":
    rows = []
    for method, runs in [("guardgp", run_guardgp_nab()),
                         ("spot_q1e-4", run_spot_nab(1e-4))]:
        for sp, det, is_out, start in runs:
            pp, pr, pf1, n_det, n_ev, hit = prf_point_event(det, is_out, start)
            wp, wr, wf1, n_dev, n_fp, n_win, n_hit = prf_window(det, is_out, start)
            rows.append(dict(
                method=method, stream=sp.split("/")[-1],
                point_prec=100 * pp, point_rec=100 * pr, point_f1=100 * pf1,
                win_prec=100 * wp, win_rec=100 * wr, win_f1=100 * wf1,
                n_det_points=n_det, n_det_events=n_dev,
                n_windows=n_win, windows_hit=n_hit,
            ))
            print(f"{method:12s} {sp.split('/')[-1]:42s} "
                  f"point P/R/F1 = {100*pp:5.1f}/{100*pr:5.1f}/{100*pf1:5.1f} | "
                  f"window P/R/F1 = {100*wp:5.1f}/{100*wr:5.1f}/{100*wf1:5.1f}",
                  flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPO, "results", "nab_window_eval.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")
