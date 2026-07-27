#!/usr/bin/env python3
"""Run the paper's GuardGP Franka pipeline (models/new_franka.py) headless and
save per-joint metrics to results/guardgp_franka.csv.

Uses the exact configuration from new_franka.py __main__ (clean_prefix=1100,
seed_min=60, shard_cap_main=100, cap_critic=20, no periodic re-optimization).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")
sys.path.insert(0, MODELS)
os.chdir(MODELS)  # new_franka.py uses relative CSV paths + flat imports

import pandas as pd
import torch

from new_franka import run_7csv_sameX_tau1to7_agents

X_COLS = [
    "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5",
    "joint_pos_6", "joint_pos_7",
    "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
    "joint_vel_6", "joint_vel_7",
]
CSV_PATHS = [f"Franka_labeled/franka{k}_labeled.csv" for k in range(1, 8)]

out = run_7csv_sameX_tau1to7_agents(
    csv_paths=CSV_PATHS,
    x_cols=X_COLS,
    time_col="Time (s)",
    clean_prefix=1100,
    device_str="cpu",
    dtype=torch.float64,
    verbose=True,
    plot_each_tau=False,
    print_each=False,
    print_every=1,
    agent_kwargs=dict(
        seed_min=60,
        shard_cap_main=100,
        cap_critic=20,
        opt_every_steps=10**9,
        opt_epochs_main=0,
        opt_epochs_crit=0,
    ),
    out_csv=os.path.join(REPO, "results", "guardgp_franka_raw.csv"),
)

rows = []
for i, r in enumerate(out["results"], 1):
    rows.append(dict(
        method="guardgp", dataset="franka", stream=f"franka{i}_labeled.csv",
        rmse=r["rmse"], nll=r["nll"], msll=r["msll"],
        precision=r["precision"], recall=r["recall"], f1=r["f1"],
        tp=r["tp"], fp=r["fp"], fn=r["fn"], tn=r["tn"],
        avg_step_ms=r["avg_step_ms"],
        route_main=r["route_main"], route_critic=r["route_critic"],
        alarms=r["alarms"],
    ))
df = pd.DataFrame(rows)
out_path = os.path.join(REPO, "results", "guardgp_franka.csv")
df.to_csv(out_path, index=False)
print(df.round(4).to_string())
print(f"\nsaved {out_path}")
