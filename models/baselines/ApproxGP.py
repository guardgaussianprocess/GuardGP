import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, time, math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func

EPS = 1e-12

def _rff_features(t_scaled: float, w: np.ndarray):
    wt = w * t_scaled
    return np.sqrt(2.0 / w.size) * np.concatenate([np.cos(wt), np.sin(wt)], axis=0)

@dataclass
class RFFCfg:
    D_rff: int = 300
    lengthscale: float = 1.0
    noise_var: float = 0.2**2
    prior_var: float = 10.0
    half_life: float = 10.0
    alpha_min: float = 0.98
    z_threshold: float = 2.576
    include_bias: bool = True
    include_linear: bool = True
    random_state: int = 0

class ApproxGPOnline:
    
    def __init__(self, cfg: RFFCfg, x_mean: float, x_std: float):
        self.cfg = cfg
        self.x_mean, self.x_std = float(x_mean), float(max(x_std, 1e-6))
        rs = np.random.RandomState(cfg.random_state)
        self.w = rs.normal(0.0, 1.0 / cfg.lengthscale, size=(cfg.D_rff,))
        self.J = None
        self.P = None
        self.prev_x = None

    
    def _phi(self, x_raw: float):
        x = (x_raw - self.x_mean) / self.x_std
        feats = []
        if self.cfg.include_bias: feats.append(1.0)
        if self.cfg.include_linear: feats.append(x)
        feats.append(_rff_features(x, self.w))
        return np.concatenate([np.atleast_1d(f) for f in feats], axis=0)

    
    def _ensure(self, d: int):
        if self.P is None:
            self.P = (1.0 / self.cfg.prior_var) * np.eye(d, dtype=float)
            self.J = np.zeros((d,), dtype=float)

    
    def _alpha_dt(self, x_raw: float):
        if self.prev_x is None: return 1.0
        dt = abs(float(x_raw) - float(self.prev_x))
        H = max(float(self.cfg.half_life), 1e-9)
        a = 2.0 ** (- dt / H)
        return float(np.clip(a, self.cfg.alpha_min, 1.0))

    
    def predict(self, x_raw: float):
        x = self._phi(x_raw)
        self._ensure(x.size)
        # 论文：预测时 solve 一次
        mu = float(x @ np.linalg.solve(self.P, self.J))
        var = float(x @ np.linalg.solve(self.P, x)) + self.cfg.noise_var
        return mu, max(var, 1e-12)

    
    def _decay_only(self, alpha: float):
        self.P *= alpha
        self.J *= alpha

   
    def update_once(self, x_raw: float, y: float, force_alpha=None, use_obs=True):
        x = self._phi(x_raw)
        self._ensure(x.size)
        alpha = self._alpha_dt(x_raw) if force_alpha is None else float(np.clip(force_alpha, self.cfg.alpha_min, 1.0))
        if not use_obs:
            self._decay_only(alpha)
        else:
            w_new = (1.0 - alpha) / self.cfg.noise_var
            self.J = alpha * self.J + w_new * (y * x)
            self.P = alpha * self.P + w_new * np.outer(x, x)
            self.P = 0.5 * (self.P + self.P.T)
        self.prev_x = float(x_raw)

    
    def step_once_timed(self, x_raw: float, y: float, z_thr: float):
        t0 = time.perf_counter()
        mu, var = self.predict(x_raw)
        t_pred = time.perf_counter() - t0

        t1 = time.perf_counter()
        z = abs(y - mu) / max(np.sqrt(var), 1e-12)
        is_anom = (z > z_thr)
        t_det = time.perf_counter() - t1

        u0 = time.perf_counter()
        if is_anom: self.update_once(x_raw, y, use_obs=False)
        else:       self.update_once(x_raw, y, use_obs=True)
        t_upd = time.perf_counter() - u0

        return mu, var, z, bool(is_anom), t_pred, t_det, t_upd, (t_pred + t_det + t_upd)


def run_approxgp_paper(
    outlier_type: str = 'uniform',
    n_total: int = 500,
    outlier_ratio: float = 0.3,
    clean_prefix: int = 50,
    seed: int = 0,
    D_rff: int = 300,
    lengthscale: float = 1.0,
    noise_var: float = 0.2**2,
    half_life: float = 10.0,
    z_threshold: float = 2.576,
    device: str = 'cpu',
    dtype: Any = torch.float64,
    verbose: bool = False,
    plot: bool = False,
    save_prefix: str = "approxgp_paper"
) -> Dict[str, Any]:
    
    torch.manual_seed(seed); np.random.seed(seed)

    
    X, y, is_outlier = make_stream_dataset_multioutlier_clean(
        n_total=n_total, outlier_ratio=outlier_ratio, outlier_type=outlier_type,
        normal_noise=np.sqrt(noise_var), x_range=(-3, 3),
        shuffle=True, shuffle_tail_only=True,
        clean_prefix=clean_prefix, random_seed=seed,
        to_tensor=True, dtype=dtype, device=device,
    )
    X_np = X.cpu().numpy().squeeze()
    y_np = y.cpu().numpy().squeeze()
    is_outlier_np = is_outlier.cpu().numpy() if isinstance(is_outlier, torch.Tensor) else np.array(is_outlier)
    clean_prefix = min(clean_prefix, X.size(0)-1)

    
    x_mean = float(np.mean(X_np[:clean_prefix]))
    x_std  = float(np.std (X_np[:clean_prefix]) + 1e-6)

    
    cfg = RFFCfg(D_rff=D_rff, lengthscale=lengthscale, noise_var=noise_var,
                 prior_var=10.0, half_life=half_life, alpha_min=0.98,
                 z_threshold=z_threshold, include_bias=True, include_linear=True,
                 random_state=seed)
    model = ApproxGPOnline(cfg, x_mean=x_mean, x_std=x_std)

    
    for i in range(clean_prefix):
        model.update_once(float(X_np[i]), float(y_np[i]), use_obs=True)

    
    f_stream = neal_func(X_np[:clean_prefix])
    mu_base_true  = float(np.mean(f_stream))
    var_base_true = float(np.var(f_stream, ddof=1)); var_base_true = max(var_base_true, EPS)
    const_base_true = 0.5 * math.log(2.0 * math.pi * var_base_true)

    
    preds, xs_pred = [], []
    abnormal_idx: List[int] = []
    t_step_list, t_pred_list, t_det_list, t_upd_list = [], [], [], []
    mse_sum = nll_sum = msll_sum = 0.0; n_pred = 0

    for t in range(clean_prefix, X.size(0)):
        x_t, y_t = float(X_np[t]), float(y_np[t])
        mu, var, z, is_anom, t_pred, t_det, t_upd, t_step = model.step_once_timed(x_t, y_t, z_threshold)
        if is_anom: abnormal_idx.append(t)

        
        y_true = float(neal_func(x_t))
        err = y_true - mu
        var_eff = max(var, EPS)
        nll_i   = 0.5*math.log(2.0*math.pi*var_eff) + 0.5*(err*err)/var_eff
        base_i  = const_base_true + 0.5*((y_true - mu_base_true)**2)/var_base_true

        mse_sum  += err*err
        nll_sum  += nll_i
        msll_sum += (nll_i - base_i)
        n_pred   += 1

        preds.append(mu); xs_pred.append(x_t)
        t_step_list.append(t_step); t_pred_list.append(t_pred)
        t_det_list.append(t_det);  t_upd_list.append(t_upd)

        if verbose and (t-clean_prefix) % 50 == 0:
            print(f"[{outlier_type}] t={t} | z={z:.2f} | anom={is_anom}")

    
    avg_mse = mse_sum / max(n_pred,1)
    rmse    = math.sqrt(max(avg_mse, 0.0))
    smse    = avg_mse / var_base_true
    avg_nll = nll_sum / max(n_pred,1)
    avg_msll= msll_sum / max(n_pred,1)
    ms = lambda L: (1000.0*np.mean(L) if L else float('nan'))
    avg_step_ms, avg_pred_ms = ms(t_step_list), ms(t_pred_list)
    avg_detect_ms, avg_upd_ms = ms(t_det_list), ms(t_upd_list)

    
    all_true_ab_idx = np.where(is_outlier_np)[0]
    stream_true_ab_idx = [idx for idx in all_true_ab_idx if idx >= clean_prefix]
    set_true, set_det = set(stream_true_ab_idx), set(abnormal_idx)
    tp = len(set_true & set_det); fp = len(set_det - set_true); fn = len(set_true - set_det)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "method": "ApproxGP-PI",
        "seed": seed,
        "outlier_type": outlier_type,
        "n_total": n_total,
        "clean_prefix": clean_prefix,
        "rmse": rmse, "smse": smse, "msll": avg_msll,
        "mse": avg_mse, "nll": avg_nll,
        "avg_step_ms": avg_step_ms, "avg_pred_ms": avg_pred_ms,
        "avg_detect_ms": avg_detect_ms, "avg_upd_ms": avg_upd_ms,
        "precision": precision, "recall": recall, "f1": f1,
        "n_pred": n_pred,
        "true_outliers": len(stream_true_ab_idx),
        "detected_outliers": len(abnormal_idx),
        "missed_outliers": len(set_true - set_det),
    }

    
    if plot:
        
        is_out = is_outlier_np.astype(bool)
        plt.figure(figsize=(10,4))
        plt.scatter(X_np[~is_out], y_np[~is_out], s=15, c='blue', label='Normal', zorder=1)
        plt.scatter(X_np[is_out],  y_np[is_out],  s=15, c='red',  label='True Outlier', zorder=1)
        if abnormal_idx:
            
            idx_detect = np.array(abnormal_idx)
            plt.scatter(X_np[idx_detect], y_np[idx_detect], s=80,
                        facecolors='none', edgecolors='black', label='Detected', linewidths=1.5, zorder=6)
        x_dense = np.linspace(np.min(X_np), np.max(X_np), 500)
        plt.plot(x_dense, neal_func(x_dense), 'k-', label='True f', zorder=2)
        if len(xs_pred)>0:
            order = np.argsort(xs_pred)
            plt.plot(np.array(xs_pred)[order], np.array(preds)[order], color='yellow', linewidth=2.5,
                     label='ApproxGP Pred', zorder=5)
        plt.xlabel('x'); plt.ylabel('y')
        plt.title(f"[{outlier_type}] ApproxGP-PI | warmup={clean_prefix} | z={z_threshold}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_prefix}_{outlier_type}_curve.png", dpi=200)
        plt.show(); plt.close()

    return result
