"""Generic single-series GuardGP runner for TSB-AD univariate streams.

Reuses the shard/critic/EVT components of models/new_NAB.py unchanged and
runs the same online loop on a plain (y, warmup_n) series: x = time index,
x and y standardized by warm-up statistics only (single-pass legit).

One default configuration everywhere = the NAB batch defaults of the paper
(shard_cap_main=100, cap_critic=20, seed_min=60, k_hist=3, alpha=0.2,
cover_frac=0.7). Returns per-point scores E_t (local-gate surprise, used for
VUS-PR/AUC) and binary alarms (route == 'alarm', used for point P/R/F1).
"""

import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS = os.path.join(_REPO, "models")
for p in (_REPO, _MODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

from new_NAB import (  # noqa: E402
    ShardManager, build_E_list, decide_label, noise_floor_of,
    build_critic_seed_from_recent, build_main_seed_cover_recent,
    greedy_farthest_points,
)
from evt_spot import SpotEVTExp, surprise_from_gaussian  # noqa: E402

EPS = 1e-12

GUARDGP_TSBAD_CFG = dict(
    shard_cap_main=100, cap_critic=20, seed_min=60,
    k_hist_main=3, alpha_main=0.2, k_hist_critic=3, alpha_critic=0.2,
    cover_frac=0.7, r_recent=100,
    k_critic_seed_recent=20, critic_seed_min=20,
    critic_rollover_inherit=True, eps_var=1e-6,
    seed_epochs=200, seed_lr=0.05,
)


def run_guardgp_series(y: np.ndarray, warmup_n: int, *, seed: int = 0,
                       cfg: dict = None, dtype=torch.float64) -> dict:
    cfg = dict(GUARDGP_TSBAD_CFG, **(cfg or {}))
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    y_np = np.asarray(y, float)
    N = len(y_np)
    assert 5 <= warmup_n < N

    # standardize x (index) and y by warm-up statistics only
    x_raw = np.arange(N, dtype=float)
    x_mean, x_std = x_raw[:warmup_n].mean(), x_raw[:warmup_n].std() + 1e-6
    y_mean, y_std = y_np[:warmup_n].mean(), y_np[:warmup_n].std() + 1e-9
    X_std = torch.tensor((x_raw - x_mean) / x_std, dtype=dtype).view(-1, 1)
    y_s = (y_np - y_mean) / y_std
    y_t = torch.tensor(y_s, dtype=dtype)

    init_idx = list(range(warmup_n))
    main_mgr = ShardManager(input_dim=1, cap=cfg["shard_cap_main"], max_hist=4,
                            device=device, dtype=dtype)
    critic_mgr = None

    X_seed0, y_seed0 = X_std[init_idx], y_t[init_idx]
    main_mgr.start_with_seed(X_seed0, y_seed0, src_gp=None,
                             epochs=cfg["seed_epochs"], lr=cfg["seed_lr"])

    K_init = max(int(cfg["seed_min"]), 1)
    K_core = min(2 * K_init, X_seed0.size(0))
    idx_core = greedy_farthest_points(X_seed0, K_core)
    X_core, y_core = X_seed0[idx_core], y_seed0[idx_core]

    pool_max = max(cfg["r_recent"], 5 * K_init)
    recent_x, recent_y = deque(maxlen=pool_max), deque(maxlen=pool_max)
    for i in init_idx:
        recent_x.append(X_std[i].detach().clone())
        recent_y.append(float(y_t[i].item()))
    critic_recent_x = deque(maxlen=cfg["r_recent"])
    critic_recent_y = deque(maxlen=cfg["r_recent"])

    E_seed = build_E_list(main_mgr.curr.gp, X_seed0, y_seed0)
    def new_evt():
        e = SpotEVTExp(q_hi=0.005, q_lo=0.02, init_quantile=0.90,
                       min_peaks=8, max_peaks=30)
        e.initialize_from_clean_E(E_seed, fallback_E_all=None, use_fallback=False)
        return e
    evt_curr, evt_global = new_evt(), new_evt()

    scores = np.zeros(N)
    alarms = np.zeros(N, dtype=bool)
    routes = {"main": 0, "critic": 0, "alarm": 0}
    w_evt_state = 0.0
    t_start = time.perf_counter()

    for t in range(warmup_n, N):
        x_t = X_std[t].view(1, -1)
        y_val = float(y_s[t])

        main_parts, critic_parts = [], []
        mu_curr_main = vy_curr_main = None
        mu_curr_crit = vy_curr_crit = None
        for shard, age in main_mgr.shards_for_fusion(cfg["k_hist_main"]):
            muM, varM, vyM = shard.predict_scalar(x_t.squeeze(0))
            if age == 0:
                mu_curr_main, vy_curr_main = muM, vyM
            w = (cfg["alpha_main"] ** age) / max(varM, cfg["eps_var"])
            main_parts.append((muM, vyM, w))

        sigma_floor_curr = max(1e-6, noise_floor_of(main_mgr.curr.gp))
        E_curr = surprise_from_gaussian(y_val, mu_curr_main, vy_curr_main,
                                        sigma_floor=sigma_floor_curr, inflate=1.3)
        lab_c, info = evt_curr.step(E_curr)
        z_hi = float(info["z_hi"]) if (info and "z_hi" in info) else None
        scores[t] = E_curr

        w_evt = w_evt_state
        if critic_mgr is not None and \
           (critic_mgr.curr is not None or len(critic_mgr.hist) > 0):
            for shard, age in critic_mgr.shards_for_fusion(cfg["k_hist_critic"]):
                muC, varC, vyC = shard.predict_scalar(x_t.squeeze(0))
                if age == 0:
                    mu_curr_crit, vy_curr_crit = muC, vyC
                w = w_evt * (cfg["alpha_critic"] ** age) / max(varC, cfg["eps_var"]) ** 0.5
                critic_parts.append((muC, vyC, w))

        parts = main_parts + critic_parts
        Wsum = sum(w for (_, _, w) in parts)
        if Wsum <= 0:
            mu_f, _, vy_f = main_mgr.curr.predict_scalar(x_t.squeeze(0))
        else:
            mu_f = sum(mu * w for (mu, _, w) in parts) / Wsum
            vy_f = sum(vy * w for (_, vy, w) in parts) / Wsum

        E_glob = None
        if lab_c == 'attack':
            if vy_curr_crit is not None:
                sigma_floor_glob = max(1e-6, min(
                    noise_floor_of(main_mgr.curr.gp),
                    noise_floor_of(critic_mgr.curr.gp)))
                v_det = min(vy_f, 1.5 * min(vy_curr_main, vy_curr_crit))
            else:
                sigma_floor_glob = max(1e-6, noise_floor_of(main_mgr.curr.gp))
                v_det = min(vy_f, 1.5 * vy_curr_main)
            E_glob = surprise_from_gaussian(y_val, mu_f, v_det,
                                            sigma_floor=sigma_floor_glob, inflate=1.3)

        final_label, route = decide_label(curr_label=lab_c, E_glob=E_glob,
                                          evt_global=evt_global)
        w_evt_state = (1.0 - math.exp(- E_curr / max(2.0 * z_hi, 1e-12))) \
            if z_hi is not None else 0.0

        routes[route] += 1
        if route == 'alarm':
            alarms[t] = True

        elif route == 'critic':
            critic_recent_x.append(X_std[t].detach().clone())
            critic_recent_y.append(y_val)
            if critic_mgr is None:
                critic_mgr = ShardManager(input_dim=1, cap=cfg["cap_critic"],
                                          max_hist=4, device=device, dtype=dtype)
            if critic_mgr.curr is None:
                if len(critic_recent_x) >= cfg["critic_seed_min"]:
                    Xc, yc = build_critic_seed_from_recent(
                        critic_recent_x, critic_recent_y, device, dtype,
                        cfg["k_critic_seed_recent"], cfg["critic_seed_min"])
                    critic_mgr.start_with_seed(Xc, yc, src_gp=None, epochs=200, lr=0.05)
            else:
                rolled_c, _ = critic_mgr.add_and_update(X_std[t], y_val)
                if rolled_c:
                    old_c = critic_mgr.hist[-1].gp if critic_mgr.hist else None
                    if len(critic_recent_x) >= cfg["critic_seed_min"]:
                        Xc, yc = build_critic_seed_from_recent(
                            critic_recent_x, critic_recent_y, device, dtype,
                            cfg["k_critic_seed_recent"], cfg["critic_seed_min"])
                    else:
                        Xc = X_std[t].view(1, -1)
                        yc = torch.tensor([y_val], dtype=dtype)
                    critic_mgr.start_with_seed(
                        Xc, yc,
                        src_gp=(old_c if cfg["critic_rollover_inherit"] else None),
                        epochs=0, lr=0.03)

        else:  # main
            rolled_m, _ = main_mgr.add_and_update(X_std[t], y_val)
            recent_x.append(X_std[t].detach().clone())
            recent_y.append(y_val)
            if rolled_m:
                old_m = main_mgr.hist[-1].gp if main_mgr.hist else None
                X_new, y_new = build_main_seed_cover_recent(
                    X_core=X_core, y_core=y_core,
                    recent_x=recent_x, recent_y=recent_y,
                    K_init=K_init, cover_frac=cfg["cover_frac"],
                    window_mult=5, device=device, dtype=dtype)
                main_mgr.start_with_seed(X_new, y_new, src_gp=old_m, epochs=0, lr=0.03)
                if old_m is not None:
                    try:
                        main_mgr.curr.gp.optimize(epochs=10, lr=0.05)
                    except Exception:
                        pass
                E_seed_new = build_E_list(main_mgr.curr.gp, X_new, y_new)
                evt_curr = SpotEVTExp(q_hi=0.005, q_lo=0.02, init_quantile=0.90,
                                      min_peaks=8, max_peaks=30)
                evt_curr.initialize_from_clean_E(E_seed_new, fallback_E_all=None,
                                                 use_fallback=False)

    total_s = time.perf_counter() - t_start
    scores[:warmup_n] = 0.0
    return dict(scores=scores, alarms=alarms, routes=routes,
                avg_step_ms=1000.0 * total_s / max(N - warmup_n, 1))
