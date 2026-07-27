import sys, os, math, time
from typing import Any, Dict, List, Tuple
from collections import deque
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from OnlineGP_Step import OnlineGP, rbf_kernel
from dataset_yahoo import make_stream_dataset_yahoo_csv
from evt_spot import SpotEVTExp, surprise_from_gaussian

EPS = 1e-12
torch.set_default_dtype(torch.float64)


# ======================== Shard + Critic Common Components ========================

def clone_hypers(src_gp, dst_gp):
    """Copy hyperparameters of old shard to new shard to ensure smooth rolling"""
    with torch.no_grad():
        if hasattr(src_gp, 'log_lengthscale') and hasattr(dst_gp, 'log_lengthscale'):
            dst_gp.log_lengthscale.data.copy_(src_gp.log_lengthscale.data)
        if hasattr(src_gp, 'log_variance') and hasattr(dst_gp, 'log_variance'):
            dst_gp.log_variance.data.copy_(src_gp.log_variance.data)
        if hasattr(src_gp, 'log_noise') and hasattr(dst_gp, 'log_noise'):
            dst_gp.log_noise.data.copy_(src_gp.log_noise.data)


def short_tune(gp, epochs=0, lr=0.02):
    """Lightweight optimization to avoid stiffness right after inheritance"""
    try:
        if epochs > 0:
            gp.optimize(epochs=epochs, lr=lr)
    except Exception as e:
        print("[warn] short_tune failed:", e)

def greedy_farthest_points(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Select k points from X (N,d) that are as dispersed as possible (coverage)
    Simple greedy: pick the point farthest from the currently selected set each time
    Returns idx (k,)
    """
    N = X.size(0)
    if k <= 0 or N == 0:
        return torch.empty(0, dtype=torch.long, device=X.device)
    if k >= N:
        return torch.arange(N, device=X.device)

    norms = (X**2).sum(dim=1)
    first = torch.argmax(norms)
    selected = [int(first.item())]

    dist2 = ((X - X[first])**2).sum(dim=1)
    for _ in range(1, k):
        nxt = torch.argmax(dist2)
        selected.append(int(nxt.item()))
        d2_new = ((X - X[nxt])**2).sum(dim=1)
        dist2 = torch.minimum(dist2, d2_new)

    return torch.tensor(selected, dtype=torch.long, device=X.device)


def build_main_seed_cover_recent(
    *,
    X_core: torch.Tensor, y_core: torch.Tensor,  # Permanent core_pool (warmup coverage subset)
    recent_x: deque, recent_y: deque,            # recent clean pool
    K_init: int,
    cover_frac: float = 0.6,
    window_mult: int = 5,
    device=None,
    dtype=torch.float64,
):
    """
    Unified seed construction (core + recent):
    candidates = core_pool + recent_tail
    cover: select K_cover from candidates using farthest-point
    recent: select K_recent only from recent_tail (backwards), excluding duplicates in cover
    """
    device = device or X_core.device
    K_init = int(K_init)

    # Target window
    W_target = max(K_init, window_mult * K_init)

    # recent_tail
    n_recent = len(recent_x)
    if n_recent > 0:
        take = min(n_recent, W_target)
        X_recent = torch.stack(list(recent_x), 0).to(device, dtype)[-take:]
        y_recent = torch.tensor(list(recent_y), device=device, dtype=dtype)[-take:]
    else:
        X_recent = X_core.new_zeros((0, X_core.size(1)))
        y_recent = y_core.new_zeros((0,))

    # candidates = core + recent_tail
    X_cand = torch.cat([X_core.to(device, dtype), X_recent], dim=0)
    y_cand = torch.cat([y_core.to(device, dtype), y_recent], dim=0)

    Nc = int(X_core.size(0))
    Nr = int(X_recent.size(0))
    W  = int(X_cand.size(0))

    # ---- cover (from both pools) ----
    K_cover = int(math.ceil(cover_frac * K_init))
    idx_cover = greedy_farthest_points(X_cand, min(K_cover, W))
    selected = set(idx_cover.tolist())

    # ---- recent (only from recent segment) ----
    K_recent = max(0, K_init - int(idx_cover.numel()))
    idx_recent_list = []
    if K_recent > 0 and Nr > 0:
        # recent segment index range in cand: [Nc, Nc+Nr-1]
        for i in range(Nc + Nr - 1, Nc - 1, -1):
            if i not in selected:
                idx_recent_list.append(i)
                selected.add(i)
            if len(idx_recent_list) >= K_recent:
                break
        idx_recent_list = list(reversed(idx_recent_list))

    idx_recent = torch.tensor(idx_recent_list, device=device, dtype=torch.long) \
        if idx_recent_list else torch.empty(0, device=device, dtype=torch.long)

    idx = torch.cat([idx_cover, idx_recent], dim=0)

    # If still less than K_init, supplement from candidates (deduplicated), prioritizing recent segment then core segment
    if idx.numel() < K_init:
        # Supplement recent first
        for i in range(Nc + Nr - 1, -1, -1):
            if i not in selected:
                idx = torch.cat([idx, torch.tensor([i], device=device, dtype=torch.long)])
                selected.add(i)
            if idx.numel() >= K_init:
                break

    return X_cand[idx[:K_init]], y_cand[idx[:K_init]]

class GPShard:
    def __init__(self, input_dim=1, device="cpu", dtype=torch.float64):
        self.gp = OnlineGP(
            input_dim=input_dim,
            init_lengthscale=1,
            init_variance=1,
            init_noise=0.2,
        )
        self.device, self.dtype = device, dtype
        self.n_points = 0

    def seed_fit(self, X_seed, y_seed, epochs=150, lr=0.05, src_gp=None):
        if src_gp is not None:
            clone_hypers(src_gp, self.gp)

        self.gp.fit(
            X_seed.to(self.device, self.dtype),
            y_seed.to(self.device, self.dtype)
        )
        self.gp.optimize(epochs=epochs, lr=lr)
        self.n_points = int(X_seed.size(0))

    @torch.no_grad()
    def predict_scalar(self, x1d):
        mu, var, vnoi = self.gp.predict_y(x1d.unsqueeze(0))
        return float(mu.item()), float(var.item()), float(vnoi.item())

    def update(self, x1d, y):
        schur = self.gp.update(x1d, float(y))
        self.n_points += 1
        return schur

    @property
    def train_x(self):
        return self.gp.train_x


class ShardManager:
    def __init__(self, input_dim=1, cap=120, max_hist=1,
                 device="cpu", dtype=torch.float64):
        self.cap = cap
        self.max_hist = max_hist
        self.device = device
        self.dtype = dtype
        self.input_dim = input_dim

        self.curr = None
        self.hist = []

    def start_with_seed(self, X_seed, y_seed, src_gp=None, **opt):
        new_shard = GPShard(self.input_dim, self.device, self.dtype)
        new_shard.seed_fit(X_seed, y_seed, src_gp=src_gp, **opt)
        self.curr = new_shard

    def add_and_update(self, x1d, y):
        schur = self.curr.update(x1d, y)
        rolled = False
        if self.curr.n_points >= self.cap:
            self.hist.append(self.curr)
            if len(self.hist) > self.max_hist:
                self.hist.pop(0)
            self.curr = None
            rolled = True
        return rolled, schur

    def shards_for_fusion(self, k_hist):
        out = []
        if self.curr is not None:
            out.append((self.curr, 0))
        for age, s in enumerate(reversed(self.hist), start=1):
            
            if age > k_hist:
                break
            out.append((s, age))
        return out


@torch.no_grad()
def build_E_list(gp: OnlineGP, Xs, ys):
    mu, var, vnoi = gp.predict_y(Xs)
    mu_l   = mu.reshape(-1).tolist()
    vnoi_l = vnoi.reshape(-1).tolist()
    ys_l   = ys.reshape(-1).tolist()
    sigma  = float(torch.exp(gp.log_noise).item())
    return [
        surprise_from_gaussian(yi, mui, vni,
                               sigma_floor=sigma,
                               inflate=1.3)
        for yi, mui, vni in zip(ys_l, mu_l, vnoi_l)
    ]


def decide_label(curr_label, E_glob=None,
                 evt_global=None):
    if curr_label == 'clean':
        return 'clean', 'main'
    if curr_label == 'uncertain':
        return 'uncertain', 'critic'

    # curr_label == 'attack'
    if (evt_global is None) or (E_glob is None):
        return 'attack', 'alarm'

    lab_g, _ = evt_global.step(E_glob)
    if lab_g != 'attack':
        return 'uncertain', 'critic'
    else:
        return 'attack', 'alarm'


def noise_floor_of(gp) -> float:
    return float(torch.exp(gp.log_noise).item())


def build_critic_seed_from_recent(critic_recent_x,
                                  critic_recent_y,
                                  device, dtype,
                                  k_critic_seed_recent: int,
                                  critic_seed_min: int):
    n_total = len(critic_recent_x)
    assert n_total >= critic_seed_min, \
        f"Insufficient points in critic_recent (< {critic_seed_min}), build_critic_seed_from_recent should not be called"

    n_take = min(k_critic_seed_recent, n_total)
    Xr = torch.stack(list(critic_recent_x), 0).to(device, dtype)[-n_take:]
    yr = torch.tensor(list(critic_recent_y),
                      device=device, dtype=dtype)[-n_take:]
    return Xr, yr


# =============================== Single Sequence: GP-EVT Shard+Critic on NAB ===============================

def run_gpevt_on_yahoo_shard_critic(
    csv_path: str,                 # (CHANGED) Direct file path
    warmup_points: int = 50,       # (CHANGED) warmup by point count
    cover_frac: float = 0.6,

    # ===== Main / Critic / Shard Hyperparameters (Keep NAB Defaults) =====
    shard_cap_main: int = 50,
    cap_critic: int = 10,
    k_anchor: int = 0,
    k_active: int = 0,
    active_margin: float = 0.0,
    r_recent: int = 100,
    k_recent: int = 50,
    seed_min: int = 1,
    k_hist_main: int = 0,
    alpha_main: float = 0.7,
    k_hist_critic: int = 0,
    alpha_critic: float = 0.7,
    eps_var: float = 1e-6,
    use_critic: bool = True,
    use_critic_in_fusion: bool = True,
    k_critic_seed_recent: int = 9,
    critic_seed_min: int = 9,
    critic_init_inherit_main: bool = False,
    critic_rollover_inherit: bool = True,
    use_proto_neal: bool = False,
    num_proto_neal: int = 20,

    # ===== EVT Switches =====
    use_evt_currmain: bool = True,
    use_evt_global: bool = True,

    # ===== Online Optimization Pacing =====
    opt_every: int = 0,
    opt_epochs_main: int = 100,
    opt_lr_main: float = 0.05,
    opt_epochs_crit: int = 100,
    opt_lr_crit: float = 0.03,

    # ===== Others =====
    seed: int = 0,
    device_str: str = "cpu",
    dtype: Any = torch.float64,
    normalize_y: bool = True,
    verbose: bool = False,
    plot: bool = False,

    # (CHANGED) Yahoo x construction strategy
    x_mode: str = "linspace",      # "linspace" or "index"
    normalize_x: bool = True,      # Whether to use warmup segment z-score for x (recommended True)

    save_prefix: str = "gpevt_yahoo_shard_critic",
    export_csv: bool = True,
) -> Dict[str, Any]:

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    # ==========================================================
    # 1) Load Yahoo CSV (CHANGED)
    # ==========================================================
    X_plot, X_feat, y, is_outlier = make_stream_dataset_yahoo_csv(
        csv_path=csv_path,
        warmup=warmup_points,
        device=device,
        dtype=dtype,
        x_mode=x_mode,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        return_norm=False,
    )
    X_plot_np = X_plot.detach().cpu().numpy().squeeze()
    X_feat_np = X_feat.detach().cpu().numpy().squeeze()
    y_np      = y.detach().cpu().numpy().squeeze()
    is_out_np = is_outlier.detach().cpu().numpy().astype(bool)

    N = len(X_feat_np)
    if N < 5:
        raise RuntimeError(f"Sequence too short to run: N={N}")

    # ==========================================================
    # 2) Warmup window: by point count (CHANGED)
    # ==========================================================
    init_len = int(max(1, min(warmup_points, N - 1)))
    init_idx = list(range(init_len))

    stream_start_idx = init_len
    if stream_start_idx >= N:
        raise RuntimeError("Insufficient data to start online stage: warmup took full sequence.")

    if verbose:
        print(f"[Init] Offline window: [{init_idx[0]}..{init_idx[-1]}], n={len(init_idx)}; "
              f"stream starting from {stream_start_idx} .")

    # ==========================================================
    # 3) x normalization: only using warmup segment statistics
    #    Note: Even if loader already normalized x, normalizing here once more won't affect execution.
    # ==========================================================
    x_mean = float(np.mean(X_feat_np[init_idx]))
    x_std  = float(np.std (X_feat_np[init_idx]) + 1e-6)
    X_std_np = (X_feat_np - x_mean) / x_std
    X_std = torch.tensor(X_std_np, device=device, dtype=dtype).view(-1, 1)
    y_t   = y.to(device=device, dtype=dtype)

    # ===================== From here below: NAB main logic =====================
    # 4) Initialize main / critic manager
    main_mgr = ShardManager(
        input_dim=1,
        cap=shard_cap_main,
        max_hist=4,
        device=device,
        dtype=dtype
    )
    critic_mgr = None

    X_seed0 = X_std[init_idx]     # (M,1)
    y_seed0 = y_t[init_idx]       # (M,)
    main_mgr.start_with_seed(X_seed0, y_seed0, src_gp=None, epochs=200, lr=0.05)


    # Recent clean (for main)
    K_init = max(int(seed_min), 1)
    
    K_core = min(2 * K_init, X_seed0.size(0))   # Recommended 2*K_init, sufficient and compact
    idx_core = greedy_farthest_points(X_seed0.to(device, dtype), K_core)
    X_core = X_seed0[idx_core].to(device, dtype)
    y_core = y_seed0[idx_core].to(device, dtype)
    pool_max = max(r_recent, 5 * K_init)   # At least enough for candidate window
    recent_x = deque(maxlen=pool_max)
    recent_y = deque(maxlen=pool_max)
    for i in init_idx:
        recent_x.append(X_std[i].detach().clone())
        recent_y.append(float(y_t[i].item()))
    if verbose:
        print(f"[Recent] Initialized {len(recent_x)} main-clean points")

    critic_recent_x = deque(maxlen=r_recent)
    critic_recent_y = deque(maxlen=r_recent)

    # EVT initialization
    E_seed_main = build_E_list(main_mgr.curr.gp, X_seed0, y_seed0)
    evt_curr = SpotEVTExp(q_hi=0.005, q_lo=0.02,
                          init_quantile=0.90, min_peaks=8, max_peaks=30) \
               if use_evt_currmain else None
    if evt_curr:
        evt_curr.initialize_from_clean_E(E_seed_main,
                                         fallback_E_all=None,
                                         use_fallback=False)

    evt_global = SpotEVTExp(q_hi=0.005, q_lo=0.02,
                            init_quantile=0.90, min_peaks=8, max_peaks=30) \
                 if use_evt_global else None
    if evt_global:
        evt_global.initialize_from_clean_E(E_seed_main,
                                           fallback_E_all=None,
                                           use_fallback=False)

    if verbose:
        print(f"[EVT init] "
              f"{'CURR tE=%.3f'   % evt_curr.tE    if evt_curr   else 'CURR=OFF'} | "
              f"{'GLOBAL tE=%.3f' % evt_global.tE  if evt_global else 'GLOBAL=OFF'} | ")

    # ----- Evaluation Statistics Containers -----
    abnormal_idx: List[int] = []
    gp1_indices: List[int] = []
    gp0_indices: List[int] = init_idx.copy()

    t_step_list, t_pred_list, t_det_list, t_upd_list = [], [], [], []
    n_pred = 0
    mse_sum = 0.0

    preds, xs_pred = [], []
    preds_main, preds_critic = [], []

    n_route_main   = 0
    n_route_critic = 0

    var_base = float(np.var(y_np[stream_start_idx:], ddof=1))
    var_base = max(var_base, EPS)

    w_evt_state = 0.0
    # ========================= Main Online Loop =========================
    for t in range(stream_start_idx, N):
        t0 = time.perf_counter()
        x_t = X_std[t].view(1, -1)
        y_t_val = float(y_np[t])

        # --- Fusion Prediction Stage ---
        t1 = time.perf_counter()

        main_parts   = []
        critic_parts = []
        W_main_raw   = 0.0
        W_crit_raw   = 0.0
        Vy_main_num  = 0.0
        Vy_crit_num  = 0.0

        mu_curr_main = None; vy_curr_main = None
        mu_curr_crit = None; vy_curr_crit = None

        # Main shards
        for shard, age in main_mgr.shards_for_fusion(k_hist_main):
            muM, varM, vyM = shard.predict_scalar(x_t.squeeze(0))

            if age == 0:
                mu_curr_main = muM
                vy_curr_main = vyM

            w_rec  = (alpha_main ** age)
            w_prec = 1.0 / max(varM, eps_var)
            w_raw  = w_rec * w_prec

            main_parts.append((muM, varM, vyM, w_raw))
            W_main_raw  += w_raw
            Vy_main_num += w_raw * vyM

        t_pred = time.perf_counter() - t1

        # --- Detection Stage: Hierarchical EVT Chain ---
        t1 = time.perf_counter()
        E_curr = None
        # 1. CurrMain EVT (sigma floor = observation noise)
        if evt_curr is not None:
            sigma_floor_curr = max(1e-6, noise_floor_of(main_mgr.curr.gp))
            E_curr = surprise_from_gaussian(
                y_t_val, mu_curr_main, vy_curr_main,
                sigma_floor=sigma_floor_curr,
                inflate=1.3
            )
            lab_c, info = evt_curr.step(E_curr)
        else:
            lab_c, info = 'uncertain', None

        if (info is not None) and ("z_lo" in info) and ("z_hi" in info):
            z_lo = float(info["z_lo"]); z_hi = float(info["z_hi"])
        else:
            z_lo, z_hi = None, None
        

        # ===== EVT-based critic gate =====
        t_det = time.perf_counter() - t1
        #if (E_curr is not None) and (z_hi is not None):
            # bounded responsibility from main model inadequacy
         #   w_evt = 1.0 - math.exp(- E_curr / max(2.0 * z_hi, 1e-12))
        #else:
         #   w_evt = 0.0
        w_evt = w_evt_state
        # Combine main + critic (scaled)
        parts = []
        Wsum   = 0.0
        Vy_num = 0.0

        t1_pred = time.perf_counter()
        # Critic shards
        if use_critic and use_critic_in_fusion and \
        (critic_mgr is not None) and \
        (critic_mgr.curr is not None or len(critic_mgr.hist) > 0):

            for shard, age in critic_mgr.shards_for_fusion(k_hist_critic):
                muC, varC, vyC = shard.predict_scalar(x_t.squeeze(0))

                if age == 0:  # Current critic
                    mu_curr_crit = muC
                    vy_curr_crit = vyC

                w_rec  = (alpha_critic ** age)
                w_prec = 1.0 / max(varC, eps_var)**0.5
                #w_prec = 1.0 / max(varC, EPS_VAR)
                w_raw  =  w_evt * w_rec * w_prec

                critic_parts.append((muC, varC, vyC, w_raw))
                W_crit_raw  += w_raw
                Vy_crit_num += w_raw * vyC

        for (muM, varM, vyM, w_raw) in main_parts:
            parts.append((muM, varM, vyM, w_raw))
            Wsum   += w_raw
            Vy_num += w_raw * vyM

        for (muC, varC, vyC, w_raw) in critic_parts:
            parts.append((muC, varC, vyC, w_raw))
            Wsum   += w_raw
            Vy_num += w_raw * vyC

        if Wsum <= 0:
            # Extreme fallback
            mu_f, v_f, vy_f = main_mgr.curr.predict_scalar(x_t.squeeze(0))
            if mu_curr_main is None or vy_curr_main is None:
                mu_curr_main, _, vy_curr_main = main_mgr.curr.predict_scalar(x_t.squeeze(0))
        else:
            mu_f = sum(mu*w for (mu, _, _, w) in parts) / Wsum
            v_f  = 1.0 / Wsum
            vy_f = Vy_num / Wsum

        t_pred += time.perf_counter() - t1_pred
        lab_g = 'NA'
        E_glob = None
        
        tdetect1 = time.perf_counter()
        if lab_c == 'attack' and use_evt_global and evt_global is not None:
            has_curr_critic = (critic_mgr is not None) and (critic_mgr.curr is not None) and (vy_curr_crit is not None)
            if has_curr_critic:
                sigma_floor_glob = max(1e-6, min(
                    noise_floor_of(main_mgr.curr.gp),
                    noise_floor_of(critic_mgr.curr.gp)
                ))
                gamma = 1.5
                v_det = min(vy_f, gamma * min(vy_curr_main, vy_curr_crit))
            else:
                sigma_floor_glob = max(1e-6, noise_floor_of(main_mgr.curr.gp))
                gamma = 1.5
                v_det = min(vy_f, gamma * vy_curr_main)

            E_glob = surprise_from_gaussian(
                y_t_val, mu_f, v_det,
                sigma_floor=sigma_floor_glob,
                inflate=1.3
            )

        final_label, route = decide_label(
            curr_label=lab_c,
            E_glob=E_glob,
            evt_global=evt_global
        )

        t_det = time.perf_counter() - tdetect1
        if (E_curr is not None) and (z_hi is not None):
            w_evt_next = 1.0 - math.exp(- E_curr / max(2.0 * z_hi, 1e-12))
        else:
            w_evt_next = 0.0
        # --- Routing and Updating ---
        t3 = time.perf_counter()
        rolled_m = False
        rolled_c = False

        if route == 'alarm':
            abnormal_idx.append(t)

        elif route == 'critic':
            gp1_indices.append(t)
            n_route_critic += 1

            critic_recent_x.append(X_std[t].detach().clone())
            critic_recent_y.append(float(y_t_val))

            if use_critic:
                if critic_mgr is None:
                    critic_mgr = ShardManager(
                        input_dim=1,
                        cap=cap_critic,
                        max_hist=1,
                        device=device,
                        dtype=dtype
                    )

                if getattr(critic_mgr, "curr", None) is None:
                    if len(critic_recent_x) >= critic_seed_min:
                        X_seed_c, y_seed_c = build_critic_seed_from_recent(
                            critic_recent_x, critic_recent_y,
                            device, dtype,
                            k_critic_seed_recent,
                            critic_seed_min
                        )
                        critic_mgr.start_with_seed(
                            X_seed_c, y_seed_c,
                            src_gp=(main_mgr.curr.gp if critic_init_inherit_main else None),
                            epochs=200, lr=0.05
                        )
                else:
                    rolled_c, _ = critic_mgr.add_and_update(X_std[t], y_t_val)

                    if rolled_c:
                        old_c_gp = critic_mgr.hist[-1].gp if len(critic_mgr.hist) > 0 else None

                        if len(critic_recent_x) >= critic_seed_min:
                            X_seed_c_new, y_seed_c_new = build_critic_seed_from_recent(
                                critic_recent_x, critic_recent_y,
                                device, dtype,
                                k_critic_seed_recent,
                                critic_seed_min
                            )
                        else:
                            X_seed_c_new = X_std[t].view(1, -1)
                            y_seed_c_new = torch.tensor([y_t_val],
                                                        device=device, dtype=dtype)

                        critic_mgr.start_with_seed(
                            X_seed_c_new, y_seed_c_new,
                            src_gp=(old_c_gp if critic_rollover_inherit else None),
                            epochs=0, lr=0.03
                        )
                        short_tune(critic_mgr.curr.gp, epochs=0, lr=0.02)

        elif route == 'main':
            rolled_m, schur_m = main_mgr.add_and_update(X_std[t], y_t_val)
            gp0_indices.append(t)
            n_route_main += 1

            recent_x.append(X_std[t].detach().clone())
            recent_y.append(float(y_t_val))


            if rolled_m:
                old_main_gp = main_mgr.hist[-1].gp if len(main_mgr.hist) > 0 else None

                K_init = max(int(seed_min), 1)

                X_seed_new, y_seed_new = build_main_seed_cover_recent(
                                X_core=X_core, y_core=y_core,
                                recent_x=recent_x, recent_y=recent_y,
                                K_init=K_init,
                                cover_frac=cover_frac,
                                window_mult=5,
                                device=device, dtype=dtype
                                )

                main_mgr.start_with_seed(
                    X_seed_new, y_seed_new,
                    src_gp=old_main_gp,
                    epochs=0, lr=0.03
                )
                if old_main_gp is not None:
                    short_tune(main_mgr.curr.gp, epochs=10, lr=0.05)

                if use_evt_currmain:
                    E_seed_new = build_E_list(main_mgr.curr.gp, X_seed_new, y_seed_new)
                    evt_curr = SpotEVTExp(q_hi=0.005, q_lo=0.02, init_quantile=0.90,
                                        min_peaks=8, max_peaks=30)
                    evt_curr.initialize_from_clean_E(E_seed_new, fallback_E_all=None, use_fallback=False)

        t_upd = time.perf_counter() - t3
        w_evt_state = w_evt_next
        # Online optimization (default opt_every=0 does not run)
        if opt_every > 0 and ((t - stream_start_idx) % opt_every == 0):
            uo = time.perf_counter()
            main_mgr.curr.gp.optimize(epochs=opt_epochs_main, lr=opt_lr_main)
            if use_critic and critic_mgr is not None and getattr(critic_mgr, "curr", None) is not None:
                critic_mgr.curr.gp.optimize(epochs=opt_epochs_crit, lr=opt_lr_crit)
            t_upd += time.perf_counter() - uo

        # Timing statistics
        t_step_list.append(time.perf_counter() - t0)
        t_pred_list.append(t_pred)
        t_det_list.append(t_det)
        t_upd_list.append(t_upd)

        # Error statistics (RMSE / sMSE)
        err = y_t_val - mu_f
        mse_sum += err * err
        n_pred += 1

        preds.append(mu_f)
        xs_pred.append(X_std_np[t])
        preds_main.append(mu_curr_main)
        preds_critic.append(mu_curr_crit if mu_curr_crit is not None else np.nan)

        if verbose and ((t - stream_start_idx) % 200 == 0):
            print(f"t={t} | y={y_t_val:.3f} | mu_f={mu_f:.3f} | "
                  f"lab_c={lab_c} | final={final_label} | route={route}")

    # 6) Point-level error metrics
    avg_mse = mse_sum / max(n_pred, 1)
    rmse    = math.sqrt(max(avg_mse, 0.0))
    smse    = avg_mse / var_base

    # 7) Detection statistics
    total_detected_points = len(abnormal_idx)
    detected_points_in_true = [i for i in abnormal_idx if is_out_np[i]]
    total_detected_points_in_true = len(detected_points_in_true)

    

    def ms(L):
        return 1000.0 * np.mean(L) if L else float("nan")

    avg_step_ms   = ms(t_step_list)
    avg_pred_ms   = ms(t_pred_list)
    avg_detect_ms = ms(t_det_list)
    avg_upd_ms    = ms(t_upd_list)

    # Debug counts
    idx_det_all_u = np.unique(np.array(abnormal_idx, dtype=int)) if abnormal_idx else np.array([], dtype=int)
    idx_true_pts  = np.where(is_out_np)[0]
    idx_det_true_mask = np.array(detected_points_in_true, dtype=int) if detected_points_in_true else np.array([], dtype=int)

    result: Dict[str, Any] = {
        "method": "GP-EVT-shard-critic",
        "dataset": csv_path,
        "seed": seed,
        "warmup_hours": warmup_points,
        "warmup_points": len(init_idx),
        "stream_start_idx": stream_start_idx,
        # Point-level error
        "rmse": rmse, "smse": smse, "mse": avg_mse,
        # Detection point statistics
        "total_detected_points": total_detected_points,
        "total_detected_points_in_true": total_detected_points_in_true,
        "detected_points_in_true": detected_points_in_true,
        # Timing
        "avg_step_ms": avg_step_ms,
        "avg_pred_ms": avg_pred_ms,
        "avg_detect_ms": avg_detect_ms,
        "avg_upd_ms": avg_upd_ms,
        # route statistics
        "route_main": n_route_main,
        "route_critic": n_route_critic,
        # Debug
        "dbg_count_det_all": int(idx_det_all_u.size),
        "dbg_count_true_pts": int(idx_true_pts.size),
        "dbg_count_det_in_true_by_mask": int(idx_det_true_mask.size),
        # Original detected points
        "detected_indices": abnormal_idx,
    }

    # 8) Plotting (consistent with baseline style, with critic route markers)
    if plot:
        plt.figure(figsize=(13, 4), dpi=130)


        # All points
        plt.scatter(X_plot_np, y_np, s=10, c="C0", label="Value", zorder=1)

        # Ground truth anomaly points
        if idx_true_pts.size > 0:
            plt.scatter(
                X_plot_np[idx_true_pts], y_np[idx_true_pts],
                s=18, c="red", label="True anomaly (point)", zorder=4
            )

        # All detected anomaly points (black circles)
        if len(idx_det_all_u) > 0:
            plt.scatter(
                X_plot_np[idx_det_all_u], y_np[idx_det_all_u],
                s=36, facecolors="none", edgecolors="black", linewidths=1.2,
                label=f"Detected (all) [n={len(idx_det_all_u)}]", zorder=5
            )

        # Detected points in ground truth (orange circles)
        if len(idx_det_true_mask) > 0:
            plt.scatter(
                X_plot_np[idx_det_true_mask], y_np[idx_det_true_mask],
                s=60, facecolors="none", edgecolors="orange", linewidths=1.8,
                label=f"Detected (in true) [n={len(idx_det_true_mask)}]", zorder=6
            )

        # Critic route (magenta x)
        if gp1_indices:
            idx_critic = np.array(sorted(list(set(gp1_indices))), dtype=int)
            plt.scatter(
                X_plot_np[idx_critic], y_np[idx_critic],
                s=50, marker='x', c='magenta', linewidths=1.5,
                label="Critic route", zorder=7
            )

        # Warmup segment
        plt.axvspan(
            X_plot_np[init_idx[0]], X_plot_np[init_idx[-1]],
            color="green", alpha=0.10, label="Offline warmup"
        )

        plt.xlabel("time (hours)")
        plt.ylabel("value (normalized)" if normalize_y else "value")
        plt.title(f"NAB | {csv_path} | GP-EVT-shard-critic | warmup={warmup_points}h")
        plt.legend()
        plt.tight_layout()

        base = os.path.basename(csv_path).replace(".csv", "")
        fig_path = f"{save_prefix}_{base}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        result["plot_file"] = fig_path

    # 9) Export CSV (Ground truth ranges + detected ranges in ground truth)
    if export_csv:
        import csv
        base = os.path.basename(csv_path).replace(".csv", "")
        csv_path = f"{save_prefix}_{base}_ranges.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "start_idx", "end_idx"])
        result["ranges_csv"] = csv_path

    return result
from pathlib import Path


def run_batch_gpevt_on_yahoo_shard_critic(
    
    subpaths: List[str] = None,
    *,
    fallback_warmup_points: int = 50,

    # Shard+Critic hyperparameters can be passed as needed, default uses Neal set
    cover_frac: float = 0.6,
    shard_cap_main: int = 100,
    cap_critic: int = 20,
    k_anchor: int = 0,
    k_active: int = 0,
    active_margin: float = 0.0,
    r_recent: int = 100,
    k_recent: int = 0,
    seed_min: int = 60,
    k_hist_main: int = 3,
    alpha_main: float = 0.2,
    k_hist_critic: int = 3,
    alpha_critic: float = 0.2,
    eps_var: float = 1e-6,
    use_critic: bool = True,
    use_critic_in_fusion: bool = True,
    critic_low_weight: float = 1.0,
    msll_safety_mode: bool = False,
    critic_msll_max_gain: float = 0.2,
    k_critic_seed_recent: int = 20,
    critic_seed_min: int = 20,
    critic_init_inherit_main: bool = False,
    critic_rollover_inherit: bool = True,
    use_proto_neal: bool = False,
    num_proto_neal: int = 30,

    use_evt_currmain: bool = True,
    use_evt_global: bool = True,

    opt_every: int = 12000,
    opt_epochs_main: int = 0,
    opt_lr_main: float = 0.05,
    opt_epochs_crit: int = 0,
    opt_lr_crit: float = 0.03,

    seed: int = 0,
    device_str: str = "cpu",
    dtype: Any = torch.float64,
    normalize_y: bool = True,
    verbose_each: bool = False,
    plot_each: bool = False,
    merge_by: str = "steps",
    max_gap_steps: int = 10,
    max_gap_hours: float = 0.5,
    export_each_csv: bool = False,
    excel_path: str = "gpevt_nab_shard_critic_summary.csv",
) -> pd.DataFrame:
    """
    Batch run multiple NAB sub-sequences and write key statistics to CSV.
    Column structure matches run_batch_gpevt_on_nab:
        dataset, seed, warmup_hours, warmup_points,
        avg_step_ms, avg_pred_ms, avg_detect_ms, avg_upd_ms,
        total_detected_points, total_detected_points_in_true
    """

    print(f"[BATCH-SHARD-CRITIC] Total {len(subpaths)} sequences, running one by one...")
    rows = []
    n_ok, n_fail = 0, 0

    for k, sp in enumerate(subpaths, 1):
        tag = f"[{k}/{len(subpaths)}] {sp}"
        try:

            # 2) Use correct Yahoo invocation method
            res = run_gpevt_on_yahoo_shard_critic(
                csv_path=sp,
                warmup_points=fallback_warmup_points, 
                cover_frac=cover_frac,  # original warmup_points
                shard_cap_main=shard_cap_main,
                cap_critic=cap_critic,
                k_anchor=k_anchor,
                k_active=k_active,
                active_margin=active_margin,
                r_recent=r_recent,
                k_recent=k_recent,
                seed_min=seed_min,
                k_hist_main=k_hist_main,
                alpha_main=alpha_main,
                k_hist_critic=k_hist_critic,
                alpha_critic=alpha_critic,
                eps_var=eps_var,
                use_critic=use_critic,
                use_critic_in_fusion=use_critic_in_fusion,
                k_critic_seed_recent=k_critic_seed_recent,
                critic_seed_min=critic_seed_min,
                critic_init_inherit_main=critic_init_inherit_main,
                critic_rollover_inherit=critic_rollover_inherit,
                use_proto_neal=use_proto_neal,
                num_proto_neal=num_proto_neal,
                use_evt_currmain=use_evt_currmain,
                use_evt_global=use_evt_global,
                opt_every=opt_every,
                opt_epochs_main=opt_epochs_main,
                opt_lr_main=opt_lr_main,
                opt_epochs_crit=opt_epochs_crit,
                opt_lr_crit=opt_lr_crit,
                seed=seed,
                device_str=device_str,
                dtype=dtype,
                normalize_y=normalize_y,
                verbose=verbose_each,
                plot=plot_each,
                save_prefix="gpevt_yahoo_shard_critic",
                export_csv=export_each_csv,
            )

            row = {
                "dataset": res.get("dataset", sp),
                "seed": res.get("seed", seed),
                "warmup_hours": res.get("warmup_hours", float("nan")),   # optional column to keep
                "warmup_points": res.get("warmup_points", None),
                "avg_step_ms": res.get("avg_step_ms", float("nan")),
                "avg_pred_ms": res.get("avg_pred_ms", float("nan")),
                "avg_detect_ms": res.get("avg_detect_ms", float("nan")),
                "avg_upd_ms": res.get("avg_upd_ms", float("nan")),
                "total_detected_points": res.get("total_detected_points", 0),
                "total_detected_points_in_true": res.get("total_detected_points_in_true", 0),
            }
            rows.append(row)
            n_ok += 1
            print(f"{tag} ✅ OK")

        except Exception as e:
            n_fail += 1
            print(f"{tag} ❌ FAIL: {e}")
          

    df = pd.DataFrame(rows, columns=[
        "dataset", "seed", "warmup_hours", "warmup_points",
        "avg_step_ms", "avg_pred_ms", "avg_detect_ms", "avg_upd_ms",
        "total_detected_points", "total_detected_points_in_true",
    ])

    df.to_csv(excel_path, index=False, encoding="utf-8")
    print(f"\n[BATCH-SHARD-CRITIC] Finished: succeeded {n_ok}, failed {n_fail}. Results written to: {excel_path}")
    return df


# ============================ Command Line Test Entry (Optional) ============================

if __name__ == "__main__":
    selected = [
        
        "TSB-UAD-Public-v2/TSB-UAD-Public-v2/YAHOO/Yahoo_A1real_16_data.csv",
        "TSB-UAD-Public-v2/TSB-UAD-Public-v2/YAHOO/Yahoo_A1real_53_data.csv",
        

    ]

    df = run_batch_gpevt_on_yahoo_shard_critic(
        subpaths=selected,
        fallback_warmup_points=100,
        seed=0,
        verbose_each=False,
        plot_each=True,
        merge_by="steps",
        max_gap_steps=10,
        max_gap_hours=0.5,
        export_each_csv=False,
        cover_frac=0.7,
        excel_path="yahoo_4Experts_alpha02.csv",
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\n=== GP-EVT SHARD+CRITIC NAB BATCH SUMMARY (head) ===")
        print(df.head())