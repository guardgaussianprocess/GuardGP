import pandas as pd
import numpy as np
import torch

EPS = 1e-12

def make_stream_dataset_yahoo_csv(
    csv_path: str,
    *,
    warmup: int = 40,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    x_mode: str = "linspace",      # "linspace" or "index"
    normalize_x: bool = True,      # Whether x is z-score normalized based on warmup period
    normalize_y: bool = True,      # Whether y is z-score normalized based on warmup period (no leakage)
    return_norm: bool = False,     # Whether to return y norm parameters (for restoration)
):
    df = pd.read_csv(csv_path)

    # --- Column name compatibility ---
    col_data_candidates  = ["Data", "data", "value", "Value", "val", "signal", "y", "Y"]
    col_label_candidates = ["Label", "label", "is_anomaly", "anomaly", "Anomaly", "is_outlier"]

    data_col  = next((c for c in col_data_candidates if c in df.columns), None)
    label_col = next((c for c in col_label_candidates if c in df.columns), None)

    if data_col is None:
        raise ValueError(f"Data column not found. Existing columns={list(df.columns)}")

    if label_col is None:
        # Fill with zeros if label column is missing
        lab_np = np.zeros(len(df), dtype=int)
    else:
        lab_np = df[label_col].astype(int).to_numpy()

    y_np = df[data_col].astype(float).to_numpy()
    is_out_np = (lab_np == 1)

    N = len(y_np)
    if N < 2:
        raise ValueError(f"Sequence too short: N={N}")

    warmup = int(max(1, min(warmup, N - 1)))

    # --- Construct x ---
    if x_mode == "linspace":
        x_np = np.linspace(-1.0, 1.0, N, dtype=float)
    elif x_mode == "index":
        x_np = np.arange(N, dtype=float)
    else:
        raise ValueError("x_mode must be 'linspace' or 'index'")

    # --- x: z-score using warmup period statistics only ---
    if normalize_x:
        x_mu = float(np.mean(x_np[:warmup]))
        x_sd = float(np.std(x_np[:warmup]) + EPS)
        x_np = (x_np - x_mu) / x_sd
    else:
        x_mu, x_sd = 0.0, 1.0

    # --- y: z-score using warmup period statistics only (no leakage) ---
    if normalize_y:
        y_mu = float(np.mean(y_np[:warmup]))
        y_sd = float(np.std(y_np[:warmup]) + EPS)
        y_z  = (y_np - y_mu) / y_sd
    else:
        y_mu, y_sd = 0.0, 1.0
        y_z = y_np

    device = torch.device(device)

    # Plotting: still use original index (or timestamps if preferred)
    X_plot = torch.arange(N, device=device, dtype=dtype)          # [N]
    X_feat = torch.tensor(x_np, device=device, dtype=dtype).view(-1, 1)  # [N,1]
    y      = torch.tensor(y_z,  device=device, dtype=dtype).view(-1)     # [N]
    is_out = torch.tensor(is_out_np, device=device, dtype=torch.bool)    # [N] bool

    if return_norm:
        norm = {
            "warmup": warmup,
            "x_mu": float(x_mu), "x_sd": float(x_sd),
            "y_mu": float(y_mu), "y_sd": float(y_sd),
        }
        return X_plot, X_feat, y, is_out, norm

    return X_plot, X_feat, y, is_out

