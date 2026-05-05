from typing import Any, Dict, List, Tuple
import math
import time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func
from models.evt_spot import SpotEVTExp, surprise_from_gaussian

from models.tools import (
    EPS,
    ShardManager,
    greedy_farthest_points,
    build_main_seed_cover_recent,
    build_E_list,
    decide_label,
    noise_floor_of,
    build_critic_seed_from_recent,
    short_tune,
)

def run_guardgp(
    *,
    
    n_total: int = 500,
    clean_prefix: int = 50,
    outlier_ratio: float = 0.4,
    outlier_type: str = "asymmetric",
    seed: int = 409,
    normal_noise: float = 0.2,
    x_range: Tuple[float, float] = (-3, 3),
    shuffle: bool = True,
    shuffle_tail_only: bool = True,
    cover_frac: float = 0.7,

    
    shard_cap_main: int = 50,
    cap_critic: int = 10,
    r_recent: int = 100,
    seed_min: int = 1,

    
    k_hist_main: int = 3,
    alpha_main: float = 0.2,
    k_hist_critic: int = 3,
    alpha_critic: float = 0.2,
    eps_var: float = 1e-6,

    
    use_critic: bool = True,
    use_critic_in_fusion: bool = True,
    k_critic_seed_recent: int = 5,
    critic_seed_min: int = 10,
    critic_init_inherit_main: bool = False,
    critic_rollover_inherit: bool = True,

    
    use_evt_currmain: bool = True,
    use_evt_global: bool = True,

    
    opt_every_steps: int = 1000,
    opt_epochs_main: int = 100,
    opt_lr_main: float = 0.05,
    opt_epochs_crit: int = 100,
    opt_lr_crit: float = 0.03,

   
    device_str: str = "cpu",
    dtype: Any = torch.float64,
    verbose: bool = False,
    plot: bool = False,
) -> Dict[str, Any]:
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    otype = outlier_type

    X, y, is_outlier = make_stream_dataset_multioutlier_clean(
        n_total=n_total,
        outlier_ratio=outlier_ratio,
        outlier_type=otype,
        normal_noise=normal_noise,
        x_range=x_range,
        shuffle=shuffle,
        shuffle_tail_only=shuffle_tail_only,
        clean_prefix=clean_prefix,
        random_seed=seed,
        to_tensor=True,
        dtype=dtype,
        device=device,
    )

    X_np = X.cpu().numpy().squeeze()
    y_np = y.cpu().numpy().squeeze()
    is_out_np = is_outlier.cpu().numpy().astype(bool)

    init_N = clean_prefix

  
    main_mgr = ShardManager(
        input_dim=1,
        cap=shard_cap_main,
        max_hist=4,
        device=device,
        dtype=dtype
    )
    critic_mgr = None  

    X_seed0 = X[:init_N]
    y_seed0 = y[:init_N]

    main_mgr.start_with_seed(X_seed0, y_seed0, src_gp=None, epochs=200, lr=0.05)


    K_init = max(int(seed_min), 1)
    
    K_core = min(2 * K_init, X_seed0.size(0))   
    idx_core = greedy_farthest_points(X_seed0.to(device, dtype), K_core)
    X_core = X_seed0[idx_core].to(device, dtype)
    y_core = y_seed0[idx_core].to(device, dtype)
    pool_max = max(r_recent, 5 * K_init)   
    recent_x = deque(maxlen=pool_max)
    recent_y = deque(maxlen=pool_max)

    for i in range(init_N):
        recent_x.append(X[i].detach().clone())
        recent_y.append(float(y[i].item()))
    

    
    critic_recent_x = deque(maxlen=r_recent)
    critic_recent_y = deque(maxlen=r_recent)


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

    
    abnormal_indices = []
    gp1_indices      = []
    gp0_indices      = list(range(init_N))

    t_step_list, t_pred_list, t_det_list, t_upd_list = [], [], [], []

    mse_sum = nll_sum = msll_sum = 0.0
    n_pred  = 0
    preds, sigs, xs_pred = [], [], []

    preds_main   = []
    preds_critic = []

    n_route_main   = 0
    n_route_critic = 0


    f_stream = neal_func(X_np[:init_N])
    mu_base_true  = float(np.mean(f_stream))
    var_base_true = float(np.var(f_stream, ddof=1)); var_base_true = max(var_base_true, EPS)
    const_base_true = 0.5 * math.log(2.0 * math.pi * var_base_true)
    
    w_evt_state = 0.0
   
    for t in range(init_N, X.size(0)):
        mu_curr_main = None
        vy_curr_main = None
        mu_curr_crit = None
        vy_curr_crit = None

        t_step_begin = time.perf_counter()
        x_t, y_t = X[t], y[t]

       
        t0_pred = time.perf_counter()
        main_parts   = []
        critic_parts = []
        W_main_raw   = 0.0
        W_crit_raw   = 0.0
        Vy_main_num  = 0.0
        Vy_crit_num  = 0.0
        # Main shards
        for shard, age in main_mgr.shards_for_fusion(k_hist_main):
            muM, varM, vyM = shard.predict_scalar(x_t)

            if age == 0:  
                mu_curr_main = muM
                vy_curr_main = vyM

            w_rec  = (alpha_main ** age)
            w_prec = 1.0 / max(varM, eps_var)
            w_raw  = w_rec * w_prec

            main_parts.append((muM, varM, vyM, w_raw))
            W_main_raw  += w_raw
            Vy_main_num += w_raw * vyM


        t_pred = time.perf_counter() - t0_pred

       
        t1 = time.perf_counter()
        E_curr = None
        
        if evt_curr is not None:
            sigma_floor_curr = max(1e-6, noise_floor_of(main_mgr.curr.gp))
            E_curr = surprise_from_gaussian(
                y_t.item(), mu_curr_main, vy_curr_main,
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
        

       # ===== EVT-based critic responsibility (theoretical) =====
        t_detect = time.perf_counter() - t1
        
        w_evt = w_evt_state
        
        
        parts = []
        Wsum   = 0.0
        Vy_num = 0.0

        t1_pred = time.perf_counter()
        # Critic shards
        if use_critic and use_critic_in_fusion and \
        (critic_mgr is not None) and \
        (critic_mgr.curr is not None or len(critic_mgr.hist) > 0):

            for shard, age in critic_mgr.shards_for_fusion(k_hist_critic):
                muC, varC, vyC = shard.predict_scalar(x_t)

                if age == 0:  # critic
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
            
            mu_f, v_f, vy_f = main_mgr.curr.predict_scalar(x_t)
            if mu_curr_main is None or vy_curr_main is None:
                mu_curr_main, _, vy_curr_main = main_mgr.curr.predict_scalar(x_t)
        else:
            mu_f = sum(mu*w for (mu, _, _, w) in parts) / Wsum
            v_f  = 1.0 / Wsum
            vy_f = Vy_num / Wsum

        t_pred += time.perf_counter() - t1_pred
        lab_g = 'NA'
        E_glob = None
        
        tdetect1 = time.perf_counter()
       
        if lab_c == 'attack':
            if evt_global is not None:
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
                    y_t.item(), mu_f, v_det,
                    sigma_floor=sigma_floor_glob,
                    inflate=1.3
                )

        
        final_label, route = decide_label(
            curr_label=lab_c,
            E_glob=E_glob,
            evt_global=evt_global
        )
        
        t_detect += time.perf_counter() - tdetect1

        if (E_curr is not None) and (z_hi is not None):
            w_evt_next = 1.0 - math.exp(- E_curr / max(2.0 * z_hi, 1e-12))
        else:
            w_evt_next = 0.0
        

        # update
        t_update = 0.0
        rolled_m = False
        rolled_c = False

        if route == 'alarm':
            abnormal_indices.append(t)

        elif route == 'critic':
            gp1_indices.append(t)
            n_route_critic += 1

            critic_recent_x.append(x_t.detach().clone())
            critic_recent_y.append(float(y_t.item()))

            if use_critic:
                if critic_mgr is None:
                    critic_mgr = ShardManager(
                        input_dim=1,
                        cap=cap_critic,
                        max_hist=4,
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
                    u0 = time.perf_counter()
                    rolled_c, _ = critic_mgr.add_and_update(x_t, y_t)
                    t_update += time.perf_counter() - u0

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
                            X_seed_c_new = x_t.unsqueeze(0)
                            y_seed_c_new = torch.tensor([y_t.item()],
                                                        device=device, dtype=dtype)

                        critic_mgr.start_with_seed(
                            X_seed_c_new, y_seed_c_new,
                            src_gp=(old_c_gp if critic_rollover_inherit else None),
                            epochs=0, lr=0.03
                        )
                        short_tune(critic_mgr.curr.gp, epochs=0, lr=0.02)

        elif route == 'main':
            u0 = time.perf_counter()
            rolled_m, schur_m = main_mgr.add_and_update(x_t, y_t)
            t_update += time.perf_counter() - u0
            gp0_indices.append(t)
            n_route_main += 1

            recent_x.append(x_t.detach().clone())
            recent_y.append(float(y_t.item()))

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
                    short_tune(main_mgr.curr.gp, epochs=5, lr=0.05)

                if use_evt_currmain:
                    E_seed_new = build_E_list(main_mgr.curr.gp, X_seed_new, y_seed_new)
                    evt_curr = SpotEVTExp(q_hi=0.005, q_lo=0.02, init_quantile=0.90,
                                        min_peaks=8, max_peaks=30)
                    evt_curr.initialize_from_clean_E(E_seed_new, fallback_E_all=None, use_fallback=False)

        w_evt_state = w_evt_next        

        # optimize
        if (t - init_N) % opt_every_steps == 0:
            uo = time.perf_counter()
            main_mgr.curr.gp.optimize(epochs=opt_epochs_main, lr=opt_lr_main)
            if use_critic and critic_mgr is not None and getattr(critic_mgr, "curr", None) is not None:
                critic_mgr.curr.gp.optimize(epochs=opt_epochs_crit, lr=opt_lr_crit)
            t_update += time.perf_counter() - uo

        
        t_step = time.perf_counter() - t_step_begin
        t_step_list.append(t_step)
        t_pred_list.append(t_pred)
        t_det_list.append(t_detect)
        t_upd_list.append(t_update)

       
        y_true_val = float(neal_func(X_np[t]))
        err = y_true_val - mu_f
        var_eff = max(v_f, EPS)

        mse_sum  += err*err
        nll_i     = 0.5 * math.log(2.0 * math.pi * var_eff) + 0.5 * (err*err)/var_eff
        nll_sum  += nll_i
        nll_base_i = (const_base_true
                      + 0.5 * ((y_true_val - mu_base_true)**2) / var_base_true)
        msll_sum += (nll_i - nll_base_i)
        n_pred   += 1

        preds.append(mu_f)
        sigs.append(np.sqrt(var_eff))
        xs_pred.append(X_np[t])

        preds_main.append(mu_curr_main)
        preds_critic.append(mu_curr_crit if mu_curr_crit is not None else np.nan)

        if verbose:
            print(
                f"[t={t}] y={y_t.item():.4f} "
                f"mu_main={mu_curr_main:.4f} "
                f"mu_glob={mu_f:.4f} "
                f"vy_glob={vy_f:.4e} "
                f"| CurrMain={lab_c} "
                f"-> final={final_label} route={route} "
                f"| rolled_m={rolled_m} rolled_c={rolled_c}"
            )

    avg_mse   = mse_sum / max(n_pred, 1)
    rmse      = math.sqrt(max(avg_mse, 0.0))
    smse_true = avg_mse / var_base_true
    avg_nll   = nll_sum / max(n_pred, 1)
    avg_msll  = msll_sum / max(n_pred, 1)

    avg_step  = 1000.0 * (np.mean(t_step_list) if t_step_list else float('nan'))
    avg_pred  = 1000.0 * (np.mean(t_pred_list) if t_pred_list else float('nan'))
    avg_det   = 1000.0 * (np.mean(t_det_list)  if t_det_list  else float('nan'))
    avg_upd   = 1000.0 * (np.mean(t_upd_list)  if t_upd_list  else float('nan'))

    all_true_ab_idx      = np.where(is_out_np)[0]
    stream_true_ab_idx   = [idx for idx in all_true_ab_idx if idx >= init_N]
    missed_stream_ab_idx = [idx for idx in stream_true_ab_idx if idx not in abnormal_indices]

    set_true_ab      = set(stream_true_ab_idx)
    set_detected_ab  = set(abnormal_indices)
    tp = len(set_true_ab & set_detected_ab)
    fp = len(set_detected_ab - set_true_ab)
    fn = len(set_true_ab - set_detected_ab)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f"[{otype}] RMSE (true): {rmse:.6f}")
        print(f"[{otype}] sMSE (true): {smse_true:.6f}")
        print(f"[{otype}] MSLL (true): {avg_msll:.6f}")
        print(f"[{otype}] MSE  (true): {avg_mse:.6f}")
        print(f"[{otype}] NLL  (true Gaussian): {avg_nll:.6f}")
        print(f"[{otype}] Avg Step Time (pred+detect+update): {avg_step:.2f} ms")
        print(f"[{otype}]   ├─ Predict: {avg_pred:.2f} ms")
        print(f"[{otype}]   ├─ Detect : {avg_det:.2f} ms")
        print(f"[{otype}]   └─ Update : {avg_upd:.2f} ms")
        print(f"[{otype}] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
        print(f"[{otype}] True outliers: {stream_true_ab_idx}")
        print(f"[{otype}] Detected outliers in streaming phase: {abnormal_indices}")
        print(f"[{otype}] Missed true outliers: {missed_stream_ab_idx}")
        print(f"[{otype}] route main:   {n_route_main}")
        print(f"[{otype}] route critic: {n_route_critic}")

  
    if plot:
        plt.figure(figsize=(10,4))
        plt.scatter(X_np[~is_out_np], y_np[~is_out_np],
                    s=15, c='blue', label='Normal')
        plt.scatter(X_np[is_out_np], y_np[is_out_np],
                    s=15, c='red',  label='True Outlier')

        if abnormal_indices:
            idx_detect = np.array(sorted(list(set(abnormal_indices))), dtype=int)
            plt.scatter(X_np[idx_detect], y_np[idx_detect],
                        s=80, facecolors='none',
                        edgecolors='black',
                        linewidths=1.2,
                        label='Detected (attack)')
        if gp1_indices:
            idx_critic = np.array(sorted(list(set(gp1_indices))), dtype=int)
            plt.scatter(
                X_np[idx_critic],
                y_np[idx_critic],
                s=60,
                marker='x',
                c='magenta',
                linewidths=1.2,
                label='Critic route'
            )

        x_dense = np.linspace(np.min(X_np), np.max(X_np), 500)
        y_true_curve = neal_func(x_dense)
        plt.plot(x_dense, y_true_curve, 'k-', label='True function')

        xs_pred_arr = np.array(xs_pred)
        preds_np    = np.array(preds)
        preds_main_np    = np.array(preds_main)
        preds_critic_np  = np.array(preds_critic)
        if len(xs_pred_arr) > 0:
            sort_idx = np.argsort(xs_pred_arr)
            plt.plot(xs_pred_arr[sort_idx], preds_np[sort_idx],
                     color='orange', linewidth=2,
                     label='Fused Online GP Pred')
            plt.plot(xs_pred_arr[sort_idx], preds_main_np[sort_idx],
                     linestyle='--', linewidth=1.5,
                     label='Main per-step mean')
            if not np.all(np.isnan(preds_critic_np)):
                plt.plot(xs_pred_arr[sort_idx], preds_critic_np[sort_idx],
                         linestyle='-.', linewidth=1.5,
                         label='Critic per-step mean')

        plt.axvspan(np.min(X_np[:init_N]), np.max(X_np[:init_N]),
                    alpha=0.15, color='green', label='Preheat Clean Region')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"[{otype}] Online GP (Shard Main+Critic+Recent+Active+ProtoNeal, tiered EVT)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    return dict(
        rmse=rmse,
        smse=smse_true,
        msll=avg_msll,
        nll=avg_nll,
        f1=f1_score,
        precision=precision,
        recall=recall,
        avg_step_ms=avg_step,
        avg_pred_ms=avg_pred,
        avg_detect_ms=avg_det,
        avg_upd_ms=avg_upd,
        method="GuardGP",
        seed=seed,
        outlier_type=outlier_type,
        outlier_ratio=outlier_ratio,
        n_total=n_total,
        clean_prefix=clean_prefix,
        route_main=n_route_main,
        route_critic=n_route_critic,
        detected_indices=abnormal_indices,
        true_attack_indices=stream_true_ab_idx,
        missed_attack_indices=missed_stream_ab_idx,
    )