# 🛡️ GuardGP

[NeurIPS 2026] Resilient Online Gaussian Processes with Anomaly-Aware Gates

---

## 📦 Installation

Install all required dependencies via:

    pip install -r requirements.txt

The GP-MoE baseline script additionally requires `GPy`, `autograd`, and `pymp-pypi`.

---

## 🚀 Running the Models

Open and run `demo.ipynb` to reproduce the NEAL results.

Full per-benchmark GuardGP pipelines (run from `models/`):

| Script | Benchmark |
|---|---|
| `models/new_neal.py` | NEAL (synthetic) |
| `models/new_NAB.py` | NAB (6 streams) |
| `models/new_yahoo.py` | Yahoo A1 (2 streams) |
| `models/new_franka.py` | Franka 7-DoF robot arm |

Baseline implementations are in `models/baselines/` (Approx. GP, GP-EVT,
SGP-Q, SGP-IADAM) and `models/RCGP_Online.py` (online RCGP, IMQ weights).

---

## 🔬 Rebuttal Experiments

All additional experiments reported in the author response are reproducible
from `experiments/` (run from the repository root); raw per-run results are
in `results/*.csv`.

| Script | Experiment | Result file |
|---|---|---|
| `run_multiseed.py` | RCGP vs. GuardGP on NEAL (3 types × 8 ratios × 10 streams) and Franka; fixed-stream seed-variance runs; Wilcoxon tests | `raw_runs.csv`, `summary.csv` |
| `run_guardgp_franka.py` | GuardGP on all 7 Franka joint-torque streams | `guardgp_franka.csv` |
| `spot_baseline.py` | Non-GP detector: bi-directional DSPOT (Grimshaw GPD), q-grid, NAB + Yahoo | `spot_results.csv` |
| `run_nab_window_eval.py` | Point-wise vs. window/event-based protocol on NAB (GuardGP and SPOT) | `nab_window_eval.csv` |
| `run_warmup_sensitivity.py` | Warm-up length (20/40/80) and warm-up contamination (0–30%) | `warmup_sensitivity.csv` |
| `run_freexi_sensitivity.py` | Free-ξ GPD (Grimshaw) in both gates vs. fixed ξ=0 | `freexi_sensitivity.csv` |
| `run_gpmoe_multiseed.py` | SMC GP-MoE variance across 10 model seeds on fixed streams | `gpmoe_multiseed.csv` |
| `run_capacity_and_tnoise.py` | Expert-capacity C and peak-buffer P grids; Student-t(3) nominal noise | `capacity_tnoise.csv` |

---

## 📁 Data

`dataset/` contains the exact subsets used in the paper: NEAL generator,
6 NAB streams with `combined_windows.json` / `combined_labels.json`,
2 Yahoo A1 streams (TSB-UAD version), and the 7 labeled Franka streams.

The NAB/Yahoo/Franka pipelines expect the data under `models/NAB/`,
`models/TSB-UAD-Public-v2/`, and `models/Franka_labeled/` respectively;
the Franka files are identical to `dataset/franka/`, and the NAB/Yahoo
subsets can be copied from `dataset/` or downloaded in full below.

Full external datasets are not redistributed here due to their licenses;
they can be obtained from:

- NAB: <https://github.com/numenta/NAB> (AGPL-3.0)
- TSB-UAD (Yahoo subset): <https://github.com/TheDatumOrg/TSB-UAD>
- GP-MoE reference code: <https://github.com/michaelzhang01/GPMOE>
- WISKI (online_gp): <https://github.com/wjmaddox/online_gp>

--- 

## 🚀 Running the Models

Open and run `demo.ipynb` to reproduce the results.

---
