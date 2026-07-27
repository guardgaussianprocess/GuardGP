# dataset_nab.py
# -*- coding: utf-8 -*-
"""
NAB Data Loader (Point Labels / Window Labels / Auto Detection)
- make_stream_dataset_nab_points(...)  : Uses combined_labels.json (points), optional tol_steps / tol_seconds
- make_stream_dataset_nab_windows(...) : Uses combined_windows.json (windows), optional dilate_seconds
- make_stream_dataset_nab_auto(...)    : Automatically determines whether JSON content is points or windows and executes the corresponding branch

Returns unified 4-tuple:
  X_plot : [N,1] Cumulative hours (for plotting)
  X_feat : [N,1] Intra-day minutes * 0.01 (simple periodic feature for model input)
  y      : [N,]
  is_out : [N,] bool, point-wise labels

Dependencies: numpy / pandas / torch
"""

import os
import json
from typing import List, Tuple, Union, Sequence, Optional

import numpy as np
import pandas as pd
import torch


# ========= Reading/Parsing Utilities =========

def _load_labels_dict(labels_json_path: str):
    """Read label JSON (can be points or windows). Returns an empty dict if the file does not exist."""
    if not os.path.exists(labels_json_path):
        return {}
    with open(labels_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_points_list(entry) -> bool:
    """
    Determine whether the labels corresponding to a file are a "list of points" or a "list of windows":
      - Points: ["YYYY-mm-dd HH:MM:SS", ...]
      - Windows: [[start_iso, end_iso], ...]
    Empty list is treated as "points" for unified processing.
    """
    if not entry:
        return True
    if isinstance(entry[0], str):
        return True
    if isinstance(entry[0], (list, tuple)) and len(entry[0]) == 2:
        return False
    # Treat invalid/abnormal input as points
    return True


def _to_np_datetime64_ns(ts_col) -> np.ndarray:
    """
    Convert a column of timestamps uniformly to numpy datetime64[ns].
    NAB usually uses naive timestamps (no timezone), so utc=False is used here.
    """
    ts = pd.to_datetime(ts_col, utc=False, errors="coerce")
    return ts.values.astype("datetime64[ns]")


# ========= Point Labels -> Mask =========

def _points_to_mask_by_steps(ts: np.ndarray, point_list: Sequence[str], tol_steps: int = 0) -> np.ndarray:
    """
    Map anchor point timestamps to point-wise labels with a tolerance of "tol_steps sampling points before/after".
    ts: numpy datetime64[ns] (ascending order)
    point_list: ["YYYY-mm-dd HH:MM:SS", ...]
    """
    N = len(ts)
    mask = np.zeros(N, dtype=bool)
    if not point_list or N == 0:
        return mask

    # Use second-level integer timeline for binary search to avoid ns precision boundary issues
    tsec = ts.astype("datetime64[s]").astype(np.int64)
    import bisect

    for p in point_list:
        ps = np.datetime64(p).astype("datetime64[s]").astype(np.int64)
        i = bisect.bisect_left(tsec, ps)
        candidates = []
        if i < N:
            candidates.append(i)
        if i - 1 >= 0:
            candidates.append(i - 1)
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(tsec[k] - ps))
        lo = max(0, j - int(tol_steps))
        hi = min(N, j + int(tol_steps) + 1)
        mask[lo:hi] = True
    return mask


def _points_to_mask_by_time(ts: np.ndarray, point_list: Sequence[str], tol_seconds: float = 0.0) -> np.ndarray:
    """
    Map anchor point timestamps to point-wise labels with a time tolerance of "tol_seconds seconds before/after".
    Supports float seconds (converted to ns timedelta).
    """
    N = len(ts)
    mask = np.zeros(N, dtype=bool)
    if not point_list or N == 0:
        return mask

    tol_ns = int(round(float(tol_seconds) * 1e9))
    tol = np.timedelta64(tol_ns, "ns")
    for p in point_list:
        p64 = np.datetime64(p, "ns")
        mask |= (ts >= p64 - tol) & (ts <= p64 + tol)
    return mask


# ========= Window Labels -> Mask =========

def _timestamps_to_mask(ts: np.ndarray, win_list: Sequence[Tuple[str, str]], dilate_seconds: float = 0.0) -> np.ndarray:
    """
    ts: numpy datetime64[ns] (ascending order)
    win_list: [[start_iso, end_iso], ...]
    dilate_seconds: Number of seconds to expand before/after each window (can be 0.0)
    Returns: Boolean array is_outlier (point-wise)
    """
    if not win_list or ts.size == 0:
        return np.zeros(ts.shape[0], dtype=bool)

    mask = np.zeros(ts.shape[0], dtype=bool)
    dilate_ns = int(round(float(dilate_seconds) * 1e9))
    dil = np.timedelta64(dilate_ns, "ns")
    for (s_iso, e_iso) in win_list:
        s = np.datetime64(s_iso, "ns") - dil
        e = np.datetime64(e_iso, "ns") + dil
        mask |= (ts >= s) & (ts <= e)
    return mask


# ========= X Construction (Plotting + Model Features) =========

def _build_X_plot_and_X_feat(ts_ns: np.ndarray,
                             y: np.ndarray,
                             normalize_y: bool = True):
    """
    - X_plot: Cumulative hours (for plotting) [N,1]
    - X_feat: Intra-day minutes * 0.01 (simple periodic feature) [N,1]
    - y: Optional normalization
    """
    # X_plot: Cumulative hours from the first sample
    t0_ns = ts_ns[0].astype("datetime64[ns]").astype(np.int64)
    t_ns = ts_ns.astype("datetime64[ns]").astype(np.int64)
    dt_sec = (t_ns - t0_ns) / 1e9
    x_plot = (dt_sec / 3600.0).astype(np.float32)[:, None]

    # X_feat: Convert DatetimeIndex to Series to use .dt
    ts_series = pd.Series(pd.to_datetime(ts_ns))      # <- Key fix: only Series has .dt
    midnight = ts_series.dt.normalize()
    dt_minutes = (ts_series - midnight).dt.total_seconds() / 60.0
    x_feat = (dt_minutes.to_numpy() * 0.01).astype(np.float32)[:, None]

    # y normalization
    y = y.astype(np.float64)
    if normalize_y:
        mu = float(np.mean(y))
        std = float(np.std(y)) + 1e-12
        y = (y - mu) / std

    return x_plot, x_feat, y.astype(np.float32)



def _to_tensor_or_numpy(x_plot, x_feat, y, is_outlier, to_tensor=True, device="cpu"):
    if to_tensor:
        X_plot = torch.from_numpy(x_plot).to(device)
        X_feat = torch.from_numpy(x_feat).to(device)
        y_t = torch.from_numpy(y).to(device)
        is_outlier_t = torch.from_numpy(is_outlier.astype(np.bool_)).to(device)
        return X_plot, X_feat, y_t, is_outlier_t
    else:
        return x_plot, x_feat, y, is_outlier


# ========= Public API =========

def make_stream_dataset_nab_points(
    nab_root: str = "NAB",
    subpath: str = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    labels_file: str = "labels/combined_labels.json",
    normalize_y: bool = True,
    dropna: bool = True,
    to_tensor: bool = True,
    device: str = "cpu",
    tol_steps: int = 0,
    tol_seconds: Optional[float] = None,
):
    """
    [Point Labels Version] (Anchor points + tolerance)
    - If labels_file is a window file, it automatically falls back to taking the center points of windows to use point logic (fallback strategy).
    Returns: X_plot, X_feat, y, is_outlier
    """
    csv_path = os.path.join(nab_root, "data", subpath)
    labels_path = os.path.join(nab_root, labels_file)

    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[1]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    if dropna:
        df = df.dropna(subset=[ts_col, val_col]).reset_index(drop=True)

    ts = _to_np_datetime64_ns(df[ts_col])
    y = df[val_col].astype(np.float64).to_numpy()

    raw = _load_labels_dict(labels_path).get(subpath, [])
    # Force interpretation as "points": if JSON contains windows, take the center point of each window
    if _is_points_list(raw):
        point_list = raw
    else:
        point_list = []
        for s_iso, e_iso in raw:
            s = np.datetime64(s_iso, "ns")
            e = np.datetime64(e_iso, "ns")
            mid = s + (e - s) // 2
            point_list.append(str(mid).replace("T", " "))

    if tol_seconds is not None and tol_seconds > 0:
        is_outlier = _points_to_mask_by_time(ts, point_list, tol_seconds=float(tol_seconds))
    else:
        is_outlier = _points_to_mask_by_steps(ts, point_list, tol_steps=int(tol_steps))

    x_plot, x_feat, y = _build_X_plot_and_X_feat(ts, y, normalize_y=normalize_y)
    return _to_tensor_or_numpy(x_plot, x_feat, y, is_outlier, to_tensor=to_tensor, device=device)


def make_stream_dataset_nab_windows(
    nab_root: str = "NAB",
    subpath: str = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    labels_file: str = "labels/combined_windows.json",
    normalize_y: bool = True,
    dropna: bool = True,
    to_tensor: bool = True,
    device: str = "cpu",
    dilate_seconds: float = 0.0,
):
    """
    [Window Labels Version] ([start, end] list), with optional dilation before and after windows by dilate_seconds.
    - If labels_file is a point file, it converts each point into [p, p] to use window logic (fallback strategy).
    Returns: X_plot, X_feat, y, is_outlier
    """
    csv_path = os.path.join(nab_root, "data", subpath)
    labels_path = os.path.join(nab_root, labels_file)

    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    val_col = "value" if "value" in df.columns else df.columns[1]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    if dropna:
        df = df.dropna(subset=[ts_col, val_col]).reset_index(drop=True)

    ts = _to_np_datetime64_ns(df[ts_col])
    y = df[val_col].astype(np.float64).to_numpy()

    raw = _load_labels_dict(labels_path).get(subpath, [])
    # Force interpretation as "windows": if JSON contains points, convert to [p, p]
    if _is_points_list(raw):
        win_list = [[p, p] for p in raw]
    else:
        win_list = raw

    is_outlier = _timestamps_to_mask(ts, win_list, dilate_seconds=float(dilate_seconds))

    x_plot, x_feat, y = _build_X_plot_and_X_feat(ts, y, normalize_y=normalize_y)
    return _to_tensor_or_numpy(x_plot, x_feat, y, is_outlier, to_tensor=to_tensor, device=device)


def make_stream_dataset_nab_auto(
    nab_root: str = "NAB",
    subpath: str = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    labels_file: str = "labels/combined_labels.json",   # Either works; only used here to retrieve entry and inspect structure
    normalize_y: bool = True,
    dropna: bool = True,
    to_tensor: bool = True,
    device: str = "cpu",
    # Optional parameters for points / windows respectively:
    tol_steps: int = 0,
    tol_seconds: Optional[float] = None,
    dilate_seconds: float = 0.0,
):
    """
    Automatically determine label type (points/windows) and execute corresponding logic.
    - If JSON entry is points: use tol_steps / tol_seconds (choose one)
    - If JSON entry is windows: use dilate_seconds
    Note: labels_file only determines "which JSON to fetch the subpath entry from",
          the actual execution branch is determined by the structure of the entry (points/windows).
    """
    labels_path = os.path.join(nab_root, labels_file)
    raw = _load_labels_dict(labels_path).get(subpath, [])

    if _is_points_list(raw):
        return make_stream_dataset_nab_points(
            nab_root=nab_root, subpath=subpath, labels_file=labels_file,
            normalize_y=normalize_y, dropna=dropna, to_tensor=to_tensor, device=device,
            tol_steps=tol_steps, tol_seconds=tol_seconds
        )
    else:
        return make_stream_dataset_nab_windows(
            nab_root=nab_root, subpath=subpath, labels_file=labels_file,
            normalize_y=normalize_y, dropna=dropna, to_tensor=to_tensor, device=device,
            dilate_seconds=dilate_seconds
        )


# ========= Other Helper Functions =========

def pick_initial_clean_indices(is_outlier_np: np.ndarray, init_N: int = 50) -> List[int]:
    """
    Find a continuous "normal" segment of length >= init_N in is_outlier_np (point-wise boolean),
    and return the first init_N indices of that segment. If not found, fall back to the first init_N of all normal points.
    """
    N = len(is_outlier_np)
    run, start = 0, None
    for i in range(N):
        if not is_outlier_np[i]:
            run += 1
            if run >= init_N:
                start = i - run + 1
                break
        else:
            run = 0
    if start is not None:
        return list(range(start, start + init_N))
    clean_idx = np.where(~is_outlier_np)[0]
    return clean_idx[:init_N].tolist()


# ========= Self-test Example =========
if __name__ == "__main__":
    # Modify to your actual relative data path
    subpath_demo = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

    # 1) Point labels (combined_labels.json)
    Xp, Xf, y, out = make_stream_dataset_nab_points(
        nab_root="NAB",
        subpath=subpath_demo,
        labels_file="labels/combined_labels.json",
        tol_steps=1,            # or tol_seconds=120.0
        to_tensor=False,
    )
    print("[points] shapes:", Xp.shape, Xf.shape, y.shape, out.shape, "sum_out:", out.sum())

    # 2) Window labels (combined_windows.json)
    Xp2, Xf2, y2, out2 = make_stream_dataset_nab_windows(
        nab_root="NAB",
        subpath=subpath_demo,
        labels_file="labels/combined_windows.json",
        dilate_seconds=0.0,
        to_tensor=False,
    )
    print("[windows] shapes:", Xp2.shape, Xf2.shape, y2.shape, out2.shape, "sum_out:", out2.sum())

    # 3) Auto determination (you can also point labels_file to windows, and it will auto-select based on entry structure)
    Xp3, Xf3, y3, out3 = make_stream_dataset_nab_auto(
        nab_root="NAB",
        subpath=subpath_demo,
        labels_file="labels/combined_labels.json",
        tol_steps=1,
        dilate_seconds=0.0,
        to_tensor=False,
    )
    print("[auto] shapes:", Xp3.shape, Xf3.shape, y3.shape, out3.shape, "sum_out:", out3.sum())
