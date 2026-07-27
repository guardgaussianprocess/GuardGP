#!/usr/bin/env python3
"""SPOT / DSPOT streaming anomaly detector (Siffer et al., KDD 2017) as a
non-GP baseline for the NAB and Yahoo benchmarks.

Implements:
  * GPD tail fit via Grimshaw's method (with exponential fallback),
  * SPOT: streaming peaks-over-threshold on raw values (one tail),
  * BiDSPOT: bi-directional SPOT with drift removal by a moving average of
    depth d (DSPOT), so dips and seasonal streams are handled.

Evaluation mirrors the paper's protocol exactly:
  * identical streams, identical warm-up (NAB: first 24h; Yahoo: first 100
    points), identical labels (NAB window mask from combined_windows.json,
    Yahoo native point labels),
  * point-level precision (detected points inside true regions / all
    detections) and event-level recall (true regions hit by >=1 detection),
    F1 harmonic.

Run from repo root:  python3 experiments/spot_baseline.py
Writes results/spot_results.csv (one row per stream x q).
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")
sys.path.insert(0, MODELS)

NAB_STREAMS = [
    "realTraffic/occupancy_t4013.csv",
    "realTraffic/speed_6005.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "realKnownCause/ec2_request_latency_system_failure.csv",
    "realAdExchange/exchange-3_cpc_results.csv",
]
YAHOO_STREAMS = [
    "TSB-UAD-Public-v2/TSB-UAD-Public-v2/YAHOO/Yahoo_A1real_16_data.csv",
    "TSB-UAD-Public-v2/TSB-UAD-Public-v2/YAHOO/Yahoo_A1real_53_data.csv",
]
Q_GRID = [1e-2, 1e-3, 1e-4]


# ---------------------------------------------------------------------- #
# Grimshaw GPD fit
# ---------------------------------------------------------------------- #
def _log_likelihood(peaks, gamma, sigma):
    n = peaks.size
    if gamma == 0.0:
        return -n * np.log(sigma) - peaks.sum() / sigma
    x = 1.0 + (gamma / sigma) * peaks
    if np.any(x <= 0):
        return -np.inf
    return -n * np.log(sigma) - (1.0 + 1.0 / gamma) * np.log(x).sum()


def grimshaw(peaks, epsilon=1e-8, n_grid=40):
    """Return (gamma, sigma) of the GPD fitted to `peaks` by Grimshaw's
    method: roots of w(t) = u(1+t*Y)*v(1+t*Y) - 1 over two candidate
    intervals, plus the exponential (gamma=0) candidate; pick max
    likelihood."""
    peaks = np.asarray(peaks, float)
    n = peaks.size
    Ym, YM, Ymean = peaks.min(), peaks.max(), peaks.mean()

    def w(t):
        s = 1.0 + t * peaks
        if np.any(s <= 0):
            return np.nan
        u = 1.0 + np.log(s).mean()
        v = (1.0 / s).mean()
        return u * v - 1.0

    roots = []
    a = -1.0 / YM + epsilon
    b = 2.0 * (Ymean - Ym) / (Ymean * Ym)
    c = 2.0 * (Ymean - Ym) / (Ym ** 2)
    for lo, hi in [(a, -epsilon), (epsilon, b + c)]:
        if not (hi > lo):
            continue
        ts = np.linspace(lo, hi, n_grid)
        ws = np.array([w(t) for t in ts])
        for i in range(len(ts) - 1):
            w0, w1 = ws[i], ws[i + 1]
            if np.isnan(w0) or np.isnan(w1) or w0 * w1 > 0:
                continue
            t0, t1 = ts[i], ts[i + 1]
            for _ in range(60):  # bisection
                tm = 0.5 * (t0 + t1)
                wm = w(tm)
                if np.isnan(wm):
                    break
                if w0 * wm <= 0:
                    t1 = tm
                else:
                    t0, w0 = tm, wm
            roots.append(0.5 * (t0 + t1))

    best = (0.0, max(Ymean, 1e-12))  # exponential candidate
    best_ll = _log_likelihood(peaks, 0.0, best[1])
    for t in roots:
        s = 1.0 + t * peaks
        if np.any(s <= 0):
            continue
        gamma = np.log(s).mean()
        if abs(gamma) < 1e-12 or abs(t) < 1e-12:
            continue
        sigma = gamma / t
        if sigma <= 0:
            continue
        ll = _log_likelihood(peaks, gamma, sigma)
        if ll > best_ll:
            best, best_ll = (gamma, sigma), ll
    return best


# ---------------------------------------------------------------------- #
# SPOT / DSPOT
# ---------------------------------------------------------------------- #
class SPOT:
    """One-tail streaming POT with Grimshaw GPD fit (Siffer et al. 2017)."""

    def __init__(self, q=1e-4, init_level=0.98, max_peaks=None):
        self.q = q
        self.init_level = init_level
        self.max_peaks = max_peaks
        self.t = None      # initial (peaks) threshold
        self.z = None      # decision threshold z_q
        self.peaks = None
        self.n = 0

    def initialize(self, data):
        data = np.asarray(data, float)
        self.n = data.size
        self.t = float(np.quantile(data, self.init_level))
        self.peaks = data[data > self.t] - self.t
        if self.peaks.size < 3:  # degenerate warm-up: no clear tail yet
            spread = max(float(data.std()), 1e-9)
            self.peaks = np.array([spread, 2 * spread, 3 * spread])
        self._update_z()

    def _update_z(self):
        gamma, sigma = grimshaw(self.peaks)
        Nt = self.peaks.size
        r = self.q * self.n / Nt
        if abs(gamma) < 1e-12:
            self.z = self.t + sigma * np.log(1.0 / max(r, 1e-300))
        else:
            self.z = self.t + (sigma / gamma) * (r ** (-gamma) - 1.0)

    def step(self, x):
        """Return True if x is an alarm (and excluded from the model)."""
        if x > self.z:
            return True
        self.n += 1
        if x > self.t:
            self.peaks = np.append(self.peaks, x - self.t)
            if self.max_peaks and self.peaks.size > self.max_peaks:
                self.peaks = self.peaks[-self.max_peaks:]
            self._update_z()
        return False


class BiDSPOT:
    """Bi-directional DSPOT: drift removed by a moving average of depth d,
    upper and lower tails tracked by independent SPOT instances. Alarms do
    not update the drift window (as in the DSPOT paper)."""

    def __init__(self, q=1e-4, depth=200, init_level=0.98):
        self.up = SPOT(q=q, init_level=init_level)
        self.dn = SPOT(q=q, init_level=init_level)
        self.depth = depth
        self.window = None

    def initialize(self, data):
        data = np.asarray(data, float)
        d = min(self.depth, max(len(data) // 2, 2))
        self.depth = d
        # residuals of the warm-up after local-mean removal
        res = np.array([data[i] - data[max(0, i - d):i].mean()
                        for i in range(1, len(data))])
        self.up.initialize(res)
        self.dn.initialize(-res)
        self.window = list(data[-d:])

    def step(self, x):
        m = float(np.mean(self.window))
        r = x - m
        alarm = self.up.step(r) or self.dn.step(-r)
        if not alarm:  # only normal values update the drift model
            self.window.append(x)
            if len(self.window) > self.depth:
                self.window.pop(0)
        return alarm


# ---------------------------------------------------------------------- #
# evaluation (identical protocol to the paper)
# ---------------------------------------------------------------------- #
def mask_to_ranges(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    ranges, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            ranges.append((s, p))
            s = p = i
    ranges.append((s, p))
    return ranges


def prf_point_event(det_idx, is_out, start):
    """Point-level precision, event-level recall (paper protocol)."""
    det_idx = [i for i in det_idx if i >= start]
    n_det = len(det_idx)
    det_in_true = [i for i in det_idx if is_out[i]]
    prec = len(det_in_true) / n_det if n_det else 0.0
    stream_mask = is_out.copy()
    stream_mask[:start] = False
    ranges = mask_to_ranges(stream_mask)
    hit = sum(1 for (s, e) in ranges if any(s <= i <= e for i in det_in_true))
    rec = hit / len(ranges) if ranges else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, n_det, len(ranges), hit


# ---------------------------------------------------------------------- #
def run_nab(q):
    from dataset_nab import make_stream_dataset_nab_windows
    rows = []
    cwd = os.getcwd()
    os.chdir(MODELS)
    try:
        for sp in NAB_STREAMS:
            X_plot, X_feat, y, is_out = make_stream_dataset_nab_windows(
                nab_root="NAB", subpath=sp,
                labels_file="labels/combined_windows.json",
                normalize_y=True, to_tensor=False,
            )
            y = np.asarray(y, float).ravel()
            is_out = np.asarray(is_out, bool).ravel()
            ts_h = np.asarray(X_plot, float).ravel()  # hours since start
            start = int(np.searchsorted(ts_h, ts_h[0] + 24.0))  # 24h warm-up
            steps_per_day = max(int(round(start)), 10)

            det = BiDSPOT(q=q, depth=min(steps_per_day, 300))
            det.initialize(y[:start])
            alarms = [t for t in range(start, len(y)) if det.step(y[t])]
            prec, rec, f1, n_det, n_ev, hit = prf_point_event(alarms, is_out, start)
            rows.append(dict(dataset="nab", stream=sp.split("/")[-1], q=q,
                             precision=100 * prec, recall=100 * rec, f1=100 * f1,
                             n_detected=n_det, n_events=n_ev, events_hit=hit,
                             warmup=start))
    finally:
        os.chdir(cwd)
    return rows


def run_yahoo(q):
    from dataset_yahoo import make_stream_dataset_yahoo_csv
    rows = []
    cwd = os.getcwd()
    os.chdir(MODELS)
    try:
        for sp in YAHOO_STREAMS:
            X_plot, X_feat, y, is_out = make_stream_dataset_yahoo_csv(
                sp, warmup=100, normalize_y=True,
            )
            y = y.numpy()
            is_out = is_out.numpy().astype(bool)
            start = 100
            det = BiDSPOT(q=q, depth=50)
            det.initialize(y[:start])
            alarms = [t for t in range(start, len(y)) if det.step(y[t])]
            prec, rec, f1, n_det, n_ev, hit = prf_point_event(alarms, is_out, start)
            rows.append(dict(dataset="yahoo", stream=sp.split("/")[-1], q=q,
                             precision=100 * prec, recall=100 * rec, f1=100 * f1,
                             n_detected=n_det, n_events=n_ev, events_hit=hit,
                             warmup=start))
    finally:
        os.chdir(cwd)
    return rows


if __name__ == "__main__":
    all_rows = []
    for q in Q_GRID:
        print(f"=== q = {q:g} ===", flush=True)
        all_rows += run_nab(q)
        all_rows += run_yahoo(q)
        for r in all_rows[-8:]:
            print(f"  {r['stream']:45s} P={r['precision']:6.2f} "
                  f"R={r['recall']:6.2f} F1={r['f1']:6.2f} "
                  f"(det={r['n_detected']}, events {r['events_hit']}/{r['n_events']})",
                  flush=True)
    df = pd.DataFrame(all_rows)
    out = os.path.join(REPO, "results", "spot_results.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")
