
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, gpytorch, time, math
from collections import deque
from typing import Any, Dict, List, Tuple
from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func

EPS = 1e-12
torch.set_default_dtype(torch.float64)

class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), dtype=torch.float64
        )
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)

        self.mean_module  = gpytorch.means.ConstantMean()

        rbf = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPWrapper:
    def __init__(self, input_dim=1, q=256, M=32,
                 init_lengthscale=1.0, init_variance=1.0, init_noise=0.2,
                 device="cpu"):
        self.device = device
        self.input_dim = int(input_dim)  
        self.q = int(q)
        self.M = int(M)
        self.X_buf = deque(maxlen=self.q)
        self.y_buf = deque(maxlen=self.q)

        
        Z0 = torch.linspace(0.0, 1.0, self.M, dtype=torch.float64).view(-1, 1)
        Z0 = Z0.repeat(1, self.input_dim).to(device)   # [M, input_dim]
        self.model = _SVGPModel(Z0).to(device).double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device).double()

        rbf = self.model.covar_module.base_kernel
        rbf.lengthscale = float(init_lengthscale)
        self.model.covar_module.outputscale = float(init_variance)
        self.likelihood.noise = float(init_noise)

        self._optimizer = None
        self._mll = None
        self._inducing_inited = False
        self._opt_lr = None


    def _rebuild_optimizer(self, lr=1e-2):
        
        self._optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ], lr=float(lr))
        self._opt_lr = float(lr)

    def _set_inducing_from_buffer(self):
        
        if len(self.X_buf) == 0:
            return
        X = torch.stack(list(self.X_buf)).to(self.device).double()
        n = X.size(0)

        if n < self.M:
            idx = torch.arange(n, device=self.device)
            idx = idx.repeat(int(np.ceil(self.M / max(1, n))))[:self.M]
        else:
            idx = torch.linspace(0, n - 1, self.M, device=self.device).round().long()

        Z = X[idx].detach()

        
        with torch.no_grad():
            self.model.variational_strategy.inducing_points.data.copy_(Z)

    def update(self, x, y_val):
        x = x.detach().to(self.device).double().view(1, -1).squeeze(0)
        y_t = torch.tensor(float(y_val), dtype=torch.float64, device=self.device)
        self.X_buf.append(x)
        self.y_buf.append(y_t)

    def fit(self, X, y, epochs=300, lr=0.05):
        for i in range(X.shape[0]):
            self.update(X[i], float(y[i].item()))
        # warmup 训练：这里建议初始化一次 inducing
        self.optimize(epochs=epochs, lr=lr, init_inducing="once")

    def optimize(self, epochs=100, lr=0.01, init_inducing: str = "once"):
        
        if len(self.X_buf) == 0:
            return

        X = torch.stack(list(self.X_buf)).to(self.device).double()
        Y = torch.stack(list(self.y_buf)).to(self.device).double()

        self.model.train()
        self.likelihood.train()

        
        if init_inducing == "always":
            self._set_inducing_from_buffer()
            self._inducing_inited = True
        elif init_inducing == "once":
            if not self._inducing_inited:
                self._set_inducing_from_buffer()
                self._inducing_inited = True
        elif init_inducing == "never":
            pass
        else:
            raise ValueError(f"Unknown init_inducing={init_inducing}")

        
        self._mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=len(self.X_buf)
        )

        
        if (self._optimizer is None) or (self._opt_lr is None) or (abs(self._opt_lr - float(lr)) > 1e-15):
            self._rebuild_optimizer(lr=lr)

        with gpytorch.settings.cholesky_jitter(1e-4):
            for _ in range(int(epochs)):
                self._optimizer.zero_grad()
                out = self.model(X)
                loss = -self._mll(out, Y)
                loss.backward()
                self._optimizer.step()

    @torch.no_grad()
    def predict_y(self, x):
        self.model.eval()
        self.likelihood.eval()
        x = x.view(1, -1).to(self.device).double()
        with gpytorch.settings.cholesky_jitter(1e-4):
            f_dist = self.model(x)
            y_dist = self.likelihood(f_dist)
        mu = y_dist.mean
        var_y = y_dist.variance
        var_f = f_dist.variance
        return mu.cpu(), var_f.cpu(), var_y.cpu()



def _Q_func(x):  
    return (1.0/6.0)*np.exp(-(x*x)/4.0) + 0.5*np.exp(-(x*x)/3.0)

def _q_stat(arr, W, Ws):
    a = np.array(arr, dtype=np.float64)
    mu_hat = a[-W:].mean(); var_hat = a[-W:].var(ddof=1)
    mu_til = a[-Ws:].mean()
    return (mu_til - mu_hat) / (var_hat + 1e-8)


def _choose_val_window(is_out_np: np.ndarray, start_idx: int, max_len: int = 200) -> Tuple[int, int]:
    N = len(is_out_np)
    val_s = start_idx
    val_e = min(start_idx + max_len, N - 1)
    if not is_out_np[val_s:val_e+1].any():
        ab = np.where(is_out_np[start_idx:])[0]
        if len(ab) == 0:
            return (start_idx, start_idx - 1)
        first_ab = start_idx + int(ab[0])
        e = first_ab
        while e + 1 < N and is_out_np[e + 1]:
            e += 1
        val_e = min(max(val_e, e), N - 1)
    if not is_out_np[val_s:val_e+1].any():
        return (start_idx, start_idx - 1)
    return (val_s, val_e)

def _select_eps_p_on_validation(
    X: torch.Tensor, y: torch.Tensor, is_out_np: np.ndarray,
    init_idx: List[int], val_s: int, val_e: int,
    *, svgp_ctor, q_len: int, m_induce: int,
    init_lengthscale: float, init_variance: float, noise_std: float,
    cand_quantiles=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
) -> float:
   
    temp = svgp_ctor(q=q_len, M=m_induce,
                     init_lengthscale=init_lengthscale, init_variance=init_variance,
                     init_noise=noise_std)
    for k in init_idx:
        temp.update(X[k], float(y[k].item()))
    temp.optimize(epochs=200, lr=0.05)

  
    pvals = np.zeros(val_e - val_s + 1, dtype=np.float64)
    for t in range(val_s, val_e + 1):
        with torch.no_grad():
            mu, var_f, var_y = temp.predict_y(X[t])
        mu_t = float(mu[0].item()); v_t = float(max(var_y[0].item(), EPS)); y_t = float(y[t].item())
        logp_t = -0.5*(np.log(2*np.pi*v_t) + (y_t - mu_t)**2 / v_t)
        pvals[t - val_s] = float(np.exp(max(-50.0, logp_t)))
        temp.update(X[t], y_t)  

   
    qs = np.clip(np.array(cand_quantiles)/100.0, 0.0005, 0.5)
    cand_eps = np.unique(np.quantile(pvals, qs))

    
    y_true = is_out_np[val_s:val_e+1]
    best_eps, best_f1 = cand_eps[0], -1.0
    for eps in cand_eps:
        pred = (pvals <= eps)
        tp = np.logical_and(pred, y_true).sum()
        fp = np.logical_and(pred, np.logical_not(y_true)).sum()
        fn = np.logical_and(np.logical_not(pred), y_true).sum()
        f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
        if f1 > best_f1:
            best_f1, best_eps = f1, float(eps)
    return float(best_eps)

def _select_eps_p_from_warmup(pvals: List[float], alpha: float = 0.05) -> float:
    
    pvals = np.asarray(pvals, dtype=np.float64)
    return float(np.quantile(pvals, alpha))

def _select_eps_p_on_external_validation(
    *,  
    warm_X: torch.Tensor, warm_y: torch.Tensor,
    svgp_ctor,
    q_len: int, m_induce: int,
    init_lengthscale: float, init_variance: float, noise_std: float,
    outlier_type: str, noise_std_data: float,
    dtype: Any, device: str,
    val_seed_base: int = 1000,
    val_len: int = 50,
    val_outlier_ratio: float = 0.1,
    min_pos: int = 8, min_neg: int = 30,
    max_retries: int = 10,
    cand_quantiles=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
) -> Tuple[float, Dict[str, Any]]:
    
    temp = svgp_ctor(q=q_len, M=m_induce,
                     init_lengthscale=init_lengthscale, init_variance=init_variance,
                     init_noise=noise_std)  # type: ignore
    for k in range(warm_X.shape[0]):
        temp.update(warm_X[k], float(warm_y[k].item()))
    temp.optimize(epochs=200, lr=0.05)

    
    tried = 0
    all_p, all_y = [], []
    seed_used = None
    while tried < max_retries:
        seed_v = val_seed_base + tried
        Xv, yv, is_out_v = make_stream_dataset_multioutlier_clean(
            n_total=val_len, outlier_ratio=val_outlier_ratio, outlier_type=outlier_type,
            normal_noise=noise_std_data, x_range=(-3, 3),
            shuffle=False, shuffle_tail_only=False,
            clean_prefix=0, random_seed=seed_v,
            to_tensor=True, dtype=dtype, device=device,
        )
        Xv = Xv.double(); yv = yv.double()
        is_out_np = is_out_v.cpu().numpy().astype(bool) if isinstance(is_out_v, torch.Tensor) else np.array(is_out_v, dtype=bool)
        n_pos = int(is_out_np.sum()); n_neg = int(val_len - n_pos)
        if (n_pos >= min_pos) and (n_neg >= min_neg):
            # 逐点预测，收集 p(y)
            pvals = np.zeros(val_len, dtype=np.float64)
            with torch.no_grad():
                for t in range(val_len):
                    mu, _, var_y = temp.predict_y(Xv[t])
                    mu_t = float(mu[0].item()); v_t = float(max(var_y[0].item(), EPS)); y_t = float(yv[t].item())
                    logp_t = -0.5*(np.log(2*np.pi*v_t) + (y_t - mu_t)**2 / v_t)
                    pvals[t] = float(np.exp(max(-50.0, logp_t)))
            all_p.append(pvals); all_y.append(is_out_np)
            seed_used = seed_v
            break
        tried += 1

    if len(all_p) == 0:
        
        return None, {"mode": "fallback_warmup"}

    p = np.concatenate(all_p); y_true = np.concatenate(all_y)
    qs = np.clip(np.array(cand_quantiles)/100.0, 0.0005, 0.5)
    cand_eps = np.unique(np.quantile(p, qs))
    best_eps, best_f1 = cand_eps[0], -1.0
    for eps in cand_eps:
        pred = (p <= eps)
        tp = np.logical_and(pred, y_true).sum()
        fp = np.logical_and(pred, np.logical_not(y_true)).sum()
        fn = np.logical_and(np.logical_not(pred), y_true).sum()
        f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
        tol = 1e-6
        if (f1 > best_f1 + tol) or (abs(f1 - best_f1) <= tol and eps < best_eps):
            best_f1, best_eps = f1, float(eps)

    info = {"mode": "external", "val_seed": seed_used, "val_len": int(val_len),
            "pos": int(y_true.sum()), "neg": int((~y_true).sum()), "best_f1": float(best_f1)}
    return float(best_eps), info


def run_sgpq_svgp(
   
    outlier_type: str = 'uniform',
    n_total: int = 500,
    outlier_ratio: float = 0.3,
    clean_prefix: int = 50,
    seed: int = 0,
    
    noise_std: float = 0.2,
    q_len: int = 1000,
    m_induce: int = 25,
    init_lengthscale: float = 1.0,
    init_variance: float = 1.0,
    
    alpha: float = 0.05,
    
    W: int = 125, Ws: int = 5,
    thr_E: float = 0.3, thr_L: float = 0.3,
    
    opt_every: int = 1,
    opt_epochs: int = 10,
    lr: float = 0.05,
    
    include_opt_in_update: bool = True,
   
    device: str = "cpu",
    dtype: Any = torch.float64,
    verbose: bool = False,
    plot: bool = False,
   
    val_mode: str = "external",             
    val_max_len: int = 200,                
    val_external_len: int = 50,             
    val_external_ratio: float = 0.1,       
    val_min_pos: int = 8,                   
    val_min_neg: int = 30,                  
    val_max_retries: int = 10,              
    val_seed_offset: int = 1000,            
) -> Dict[str, Any]:
    
    torch.manual_seed(seed); np.random.seed(seed)

    X, y, is_outlier = make_stream_dataset_multioutlier_clean(
        n_total=n_total, outlier_ratio=outlier_ratio, outlier_type=outlier_type,
        normal_noise=noise_std, x_range=(-3, 3),
        shuffle=True, shuffle_tail_only=True,
        clean_prefix=clean_prefix, random_seed=seed,
        to_tensor=True, dtype=dtype, device=device,
    )
    X = X.double(); y = y.double()
    X_np = X.cpu().numpy().squeeze()
    y_np = y.cpu().numpy().squeeze()
    is_outlier_np = is_outlier.cpu().numpy() if isinstance(is_outlier, torch.Tensor) else np.array(is_outlier)
    clean_prefix = min(clean_prefix, X.size(0)-1)

    
    temp_for_warm = SVGPWrapper(q=max(64, clean_prefix), M=min(m_induce, max(8, clean_prefix//2)),
                                init_lengthscale=init_lengthscale, init_variance=init_variance,
                                init_noise=noise_std, device=device)
    pvals_warm = []
    for t in range(clean_prefix):
        temp_for_warm.X_buf.clear(); temp_for_warm.y_buf.clear()
        if t > 0:
            for k in range(t):
                temp_for_warm.update(X[k], float(y[k].item()))
            temp_for_warm.optimize(epochs=200, lr=0.05)
        with torch.no_grad():
            mu, var_f, var_y = temp_for_warm.predict_y(X[t])
        mu_t = float(mu[0].item()); v_t = float(max(var_y[0].item(), EPS)); y_t = float(y[t].item());vf_t = float(max(var_f[0].item(), EPS))
        logp_t = -0.5*(np.log(2*np.pi*v_t) + (y_t - mu_t)**2 / v_t)
        pvals_warm.append(float(np.exp(max(-50.0, logp_t))))

    eps_p = None
    val_window = None
    if val_mode == "external":
        def _ctor(q, M, init_lengthscale, init_variance, init_noise):
            return SVGPWrapper(q=q, M=M,
                               init_lengthscale=init_lengthscale, init_variance=init_variance,
                               init_noise=init_noise, device=device)
        eps_info = {}
        eps_p_ext, info = _select_eps_p_on_external_validation(
            warm_X=X[:clean_prefix], warm_y=y[:clean_prefix],
            svgp_ctor=_ctor, q_len=q_len, m_induce=m_induce,
            init_lengthscale=init_lengthscale, init_variance=init_variance, noise_std=noise_std,
            outlier_type=outlier_type, noise_std_data=noise_std,
            dtype=dtype, device=device,
            val_seed_base=seed + val_seed_offset,
            val_len=val_external_len, val_outlier_ratio=val_external_ratio,
            min_pos=val_min_pos, min_neg=val_min_neg, max_retries=val_max_retries,
        )
        if eps_p_ext is None or info.get("mode") == "fallback_warmup":
            eps_p = _select_eps_p_from_warmup(pvals_warm, alpha=alpha)
            stream_start_idx = clean_prefix
        else:
            eps_p = float(eps_p_ext)
            stream_start_idx = clean_prefix
        if info.get("mode") == "external":
            val_window = [-(info["val_len"]), -1]  
        else:
            val_window = None

    elif val_mode == "inline":
        val_s, val_e = _choose_val_window(is_outlier_np, start_idx=clean_prefix, max_len=val_max_len)
        if val_e < val_s:
            eps_p = _select_eps_p_from_warmup(pvals_warm, alpha=alpha)
            stream_start_idx = clean_prefix
            val_window = None
        else:
            def _ctor(q, M, init_lengthscale, init_variance, init_noise):
                return SVGPWrapper(q=q, M=M,
                                   init_lengthscale=init_lengthscale, init_variance=init_variance,
                                   init_noise=init_noise, device=device)
            init_idx = list(range(clean_prefix))
            eps_p = _select_eps_p_on_validation(
                X=X, y=y, is_out_np=is_outlier_np, init_idx=init_idx,
                val_s=val_s, val_e=val_e,
                svgp_ctor=_ctor, q_len=q_len, m_induce=m_induce,
                init_lengthscale=init_lengthscale, init_variance=init_variance, noise_std=noise_std,
                cand_quantiles=(0.5, 1, 2, 5, 10, 20, 30, 50),
            )
            stream_start_idx = val_e + 1
            val_window = (val_s, val_e)

    else:  # "warmup_quantile"
        eps_p = _select_eps_p_from_warmup(pvals_warm, alpha=alpha)
        stream_start_idx = clean_prefix
        val_window = None

    svgp = SVGPWrapper(q=q_len, M=m_induce,
                       init_lengthscale=init_lengthscale, init_variance=init_variance,
                       init_noise=noise_std, device=device)
    for k in range(clean_prefix):
        svgp.update(X[k], float(y[k].item()))
    svgp.optimize(epochs=200, lr=0.05)

    
    f_stream = neal_func(X_np[:stream_start_idx])
    mu_base_true  = float(np.mean(f_stream)) if len(f_stream)>0 else 0.0
    var_base_true = float(np.var(f_stream, ddof=1)) if len(f_stream)>1 else 1.0
    var_base_true = max(var_base_true, EPS)
    const_base_true = 0.5 * math.log(2.0 * math.pi * var_base_true)

    
    err_hist = deque(maxlen=W)
    lik_hist = deque(maxlen=W)
    abnormal_indices: List[int] = []
    t_step_list, t_pred_list, t_detect_list, t_update_list = [], [], [], []
   
    t_opt_total = 0.0
    n_opt = 0

    mse_sum = nll_sum = msll_sum = 0.0; n_pred = 0
    preds, xs_pred, sigs = [], [], []

    for t in range(stream_start_idx, X.size(0)):
        
        t0 = time.perf_counter()

        tp0 = time.perf_counter()
        with torch.no_grad():
            mu, var_f, var_y = svgp.predict_y(X[t])
        t_pred = time.perf_counter() - tp0
        mu_t = float(mu[0].item()); v_t = float(max(var_y[0].item(), EPS)); y_t = float(y[t].item());vf_t = float(max(var_f[0].item(), EPS))

        
        td0 = time.perf_counter()
        logp_t = -0.5*(np.log(2*np.pi*v_t) + (y_t - mu_t)**2 / v_t)
        p_t = float(np.exp(max(-50.0, logp_t)))
        is_abn = (p_t <= eps_p)
        if is_abn: abnormal_indices.append(t)

        e_t = abs(y_t - mu_t)
        err_hist.append(e_t); lik_hist.append(p_t)
        QE = QL = None
        if len(err_hist) >= W and len(lik_hist) >= W:
            QE = _Q_func(_q_stat(err_hist, W, Ws))
            QL = _Q_func(_q_stat(lik_hist, W, Ws))
        t_detect = time.perf_counter() - td0

        
        tu0 = time.perf_counter()
        if not is_abn:
            svgp.update(X[t], y_t)              
        else:
            if (QE is None) or (QL is None):
                svgp.update(X[t], mu_t)         
            elif (QE <= thr_E) or (QL <= thr_L):
                svgp.update(X[t], mu_t)         
            else:
                svgp.update(X[t], y_t)          

        step = t - stream_start_idx + 1
        if (step % max(1, opt_every)) == 0:
            if include_opt_in_update:
               
                svgp.optimize(epochs=opt_epochs, lr=lr)
            else:
                
                t_opt0 = time.perf_counter()
                svgp.optimize(epochs=opt_epochs, lr=lr)
                t_opt_total += (time.perf_counter() - t_opt0)
                n_opt += 1

        t_update = time.perf_counter() - tu0

       
        y_true = float(neal_func(X_np[t]))
        err = y_true - mu_t
        var_eff = max(v_t, EPS)
        nll_i   = 0.5*math.log(2.0*math.pi*var_eff) + 0.5*(err*err)/var_eff
        base_i  = const_base_true + 0.5*((y_true - mu_base_true)**2)/var_base_true

        mse_sum  += err*err
        nll_sum  += nll_i
        msll_sum += (nll_i - base_i)
        n_pred   += 1

        preds.append(mu_t)
        xs_pred.append(X_np[t])
        sigs.append(np.sqrt(max(float(var_f[0].item()), EPS)))

        
        t_step_list.append(time.perf_counter() - t0)
        t_pred_list.append(t_pred); t_detect_list.append(t_detect); t_update_list.append(t_update)

   
    avg_mse = mse_sum / max(n_pred, 1)
    rmse    = math.sqrt(max(avg_mse, 0.0))
    smse    = avg_mse / var_base_true
    avg_nll = nll_sum / max(n_pred, 1)
    avg_msll= msll_sum / max(n_pred, 1)

    ms = lambda L: (1000.0*np.mean(L) if L else float('nan'))
    avg_step_ms, avg_pred_ms = ms(t_step_list), ms(t_pred_list)
    avg_detect_ms, avg_upd_ms = ms(t_detect_list), ms(t_update_list)

    
    if include_opt_in_update:
        avg_opt_ms_amortized = 0.0
        avg_opt_ms = 0.0
        avg_step_ms_incl_opt = avg_step_ms  
    else:
        avg_opt_ms_amortized = (1000.0 * t_opt_total) / max(n_pred, 1)
        avg_opt_ms = (1000.0 * t_opt_total / n_opt) if n_opt > 0 else 0.0
        avg_step_ms_incl_opt = avg_step_ms + avg_opt_ms_amortized

    
    all_true_ab_idx = np.where(is_outlier_np)[0]
    stream_true_ab_idx = [idx for idx in all_true_ab_idx if idx >= stream_start_idx]
    set_true, set_pred = set(stream_true_ab_idx), set(abnormal_indices)
    tp = len(set_true & set_pred); fp = len(set_pred - set_true); fn = len(set_true - set_pred)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    result = {
        "method": "SGP-Q(SVGP)",
        "seed": seed,
        "outlier_type": outlier_type,
        "n_total": n_total,
        "clean_prefix": clean_prefix,
        "stream_start_idx": int(stream_start_idx),
        "rmse": rmse, "smse": smse, "msll": avg_msll,
        "mse": avg_mse, "nll": avg_nll,
        "avg_step_ms": avg_step_ms, "avg_pred_ms": avg_pred_ms,
        "avg_detect_ms": avg_detect_ms, "avg_upd_ms": avg_upd_ms,
       
        "avg_opt_ms_amortized": avg_opt_ms_amortized,
        "avg_opt_ms": avg_opt_ms,
        "avg_step_ms_incl_opt": avg_step_ms_incl_opt,
       
        "precision": precision, "recall": recall, "f1": f1,
        "n_pred": n_pred,
        "true_outliers": len(stream_true_ab_idx),
        "detected_outliers": len(abnormal_indices),
        "missed_outliers": len(set_true - set_pred),
        "eps_p": float(eps_p),
        "val_window": None if val_window is None else list(val_window),
        "val_mode": val_mode,
        "include_opt_in_update": bool(include_opt_in_update),
    }

    
    if plot:
        import matplotlib.pyplot as plt
        is_out = is_outlier_np.astype(bool)
        plt.figure(figsize=(10,4))
        plt.scatter(X_np[~is_out], y_np[~is_out], s=15, c='blue', label='Normal')
        plt.scatter(X_np[is_out],  y_np[is_out],  s=15, c='red',  label='True Outlier')
        if len(abnormal_indices)>0:
            idx_detect = np.array(abnormal_indices)
            plt.scatter(X_np[idx_detect], y_np[idx_detect], s=80, facecolors='none', edgecolors='black',
                        label='Detected (ε_p)', linewidths=1.5)
        x_dense = np.linspace(np.min(X_np), np.max(X_np), 500)
        plt.plot(x_dense, neal_func(x_dense), 'k-', label='True function')
        if len(xs_pred)>0:
            xs = np.array(xs_pred); ys = np.array(preds); order = np.argsort(xs)
            plt.plot(xs[order], ys[order], color='orange', linewidth=2, label='SGP-Q mean')
        plt.axvline(X_np[clean_prefix], color='gray', ls='--', alpha=0.6, label='Warmup end')
        if val_window is not None:
            vs, ve = val_window
            if vs < 0:
                plt.text(0.02, 0.95, f"External val: L={-vs} (not in main stream)", transform=plt.gca().transAxes)
            else:
                plt.axvspan(X_np[vs], X_np[ve], color='purple', alpha=0.10, label='Validation (inline)')
        plt.title(f"SGP-Q (SVGP, float64) | warmup={clean_prefix} | ε_p={eps_p:.3e} | mode={val_mode}")
        plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.tight_layout(); plt.show()

    return result
