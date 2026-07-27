import math, time
import numpy as np
import torch, gpytorch
from collections import deque
from typing import Any, Dict, List, Tuple

from dataset.neal.dataset_neal import make_stream_dataset_multioutlier_clean, neal_func

EPS = 1e-12
torch.set_default_dtype(torch.float64)

class _SVGPZeroMean(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        M = inducing_points.size(0)
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(M, dtype=torch.float64)
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ZeroMean()

        rbf = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class OnlineSVGP_DT:
    
    def __init__(
        self,
        input_dim: int = 1,
        q: int = 100,
        M: int = 60,
        init_noise: float = 0.2,
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
        device: str = "cpu",
        jitter: float = 1e-4,
    ):
        self.device = device
        self.q = int(q)
        self.M = int(M)
        self.jitter = float(jitter)

        self.X_buf = deque(maxlen=self.q)
        self.y_buf = deque(maxlen=self.q)

    
        Z0 = torch.randn(self.M, input_dim, dtype=torch.float64, device=device)
        self.model = _SVGPZeroMean(Z0).to(device).double()

        # likelihood
        self.lik = gpytorch.likelihoods.GaussianLikelihood().to(device).double()
        self.lik.noise = float(init_noise)

       
        self.model.covar_module.base_kernel.lengthscale = float(init_lengthscale)
        self.model.covar_module.outputscale = float(init_variance)

        # optimizer / mll
        self._opt = None
        self._mll = None
        self._opt_lr = None

        
        self._inducing_inited = False

    def _maybe_build_opt(self, lr: float):
       
        lr = float(lr)
        if (self._opt is None) or (self._opt_lr is None) or (abs(self._opt_lr - lr) > 1e-15):
            self._opt = torch.optim.Adam(
                [{"params": self.model.parameters()}, {"params": self.lik.parameters()}],
                lr=lr,
            )
            self._opt_lr = lr

    def _refresh_mll(self):
        
        self._mll = gpytorch.mlls.VariationalELBO(self.lik, self.model, num_data=len(self.X_buf))

    def _set_inducing_from_DT_random_inplace(self):
       
        if len(self.X_buf) == 0:
            return
        X = torch.stack(list(self.X_buf)).to(self.device).double()
        n = X.size(0)

        
        if n < self.M:
            idx = torch.randint(low=0, high=n, size=(self.M,), device=self.device)
        else:
            idx = torch.randint(low=0, high=n, size=(self.M,), device=self.device)

        Z = X[idx].detach()

        with torch.no_grad():
            self.model.variational_strategy.inducing_points.data.copy_(Z)

        self._inducing_inited = True

    def push(self, x: torch.Tensor, y: float):
        x = x.detach().to(self.device).double().view(1, -1).squeeze(0)
        y = torch.tensor(float(y), dtype=torch.float64, device=self.device)
        self.X_buf.append(x)
        self.y_buf.append(y)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        
        self.model.eval(); self.lik.eval()
        x = x.to(self.device).double().view(1, -1)
        with gpytorch.settings.cholesky_jitter(self.jitter):
            f = self.model(x)
            ydist = self.lik(f)
        mu = float(ydist.mean.item())
        var_y = float(max(ydist.variance.item(), EPS))
        return mu, math.sqrt(var_y)

    def train_on_DT(
        self,
        iters: int,
        lr: float,
        reset_inducing: bool = False,
        init_inducing: str = "once",   
    ):
        if len(self.X_buf) == 0 or iters <= 0:
            return

        X = torch.stack(list(self.X_buf)).to(self.device).double()
        Y = torch.stack(list(self.y_buf)).to(self.device).double()

        self.model.train(); self.lik.train()

        
        if reset_inducing:
            
            self._set_inducing_from_DT_random_inplace()
        else:
            if init_inducing == "always":
                self._set_inducing_from_DT_random_inplace()
            elif init_inducing == "once":
                if not self._inducing_inited:
                    self._set_inducing_from_DT_random_inplace()
            elif init_inducing == "never":
                pass
            else:
                raise ValueError(f"Unknown init_inducing={init_inducing}")

        
        self._maybe_build_opt(lr=lr)
        self._refresh_mll()

        with gpytorch.settings.cholesky_jitter(self.jitter):
            for _ in range(int(iters)):
                self._opt.zero_grad()
                out = self.model(X)
                loss = -self._mll(out, Y)
                loss.backward()
                self._opt.step()


# =========================
# IADAM beta: beta(y)=Phi(1.96 - |mu-y|/sigma)
# =========================
def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def beta_iadam(mu: float, y: float, sigma: float) -> float:
    sigma = max(float(sigma), EPS)
    return _Phi(1.96 - abs(mu - y) / sigma)


def run_sgp_iadam_svgp(
    outlier_type: str = "uniform",
    n_total: int = 500,
    outlier_ratio: float = 0.3,
    clean_prefix: int = 50,
    seed: int = 0,
    noise_std: float = 0.2,

    # DT/inducing
    q: int = 100,
    M: int = 60,

    
    init_lengthscale: float = 1.0,
    init_variance: float = 1.0,

    # IADAM
    beta_max: float = 0.05,

    
    warmup_iters: int = 1000,
    opt_every: int = 200,
    opt_epochs: int = 100,
    lr: float = 0.05,

    reset_inducing_each_train: bool = False,
    include_opt_in_update: bool = True,

    
    init_inducing: str = "once",   # "once" | "always" | "never"

    device: str = "cpu",
    dtype: Any = torch.float64,
    verbose: bool = False,
) -> Dict[str, Any]:

    torch.manual_seed(seed); np.random.seed(seed)

    X, y, is_out = make_stream_dataset_multioutlier_clean(
        n_total=n_total, outlier_ratio=outlier_ratio, outlier_type=outlier_type,
        normal_noise=noise_std, x_range=(-3, 3),
        shuffle=True, shuffle_tail_only=True,
        clean_prefix=clean_prefix, random_seed=seed,
        to_tensor=True, dtype=dtype, device=device,
    )
    X = X.double(); y = y.double()
    X_np = X.cpu().numpy().squeeze()
    is_out_np = is_out.cpu().numpy().astype(bool)

    clean_prefix = min(int(clean_prefix), n_total - 1)
    stream_start_idx = clean_prefix

    q_eff = max(1, int(q))
    M_eff = min(int(M), max(4, q_eff))

    svgp = OnlineSVGP_DT(
        q=q_eff, M=M_eff,
        init_noise=noise_std,
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
        device=device,
    )

    
    for t in range(clean_prefix):
        svgp.push(X[t], float(y[t].item()))

   
    if clean_prefix > 0 and warmup_iters > 0:
        svgp.train_on_DT(
            iters=warmup_iters, lr=lr,
            reset_inducing=reset_inducing_each_train,
            init_inducing=init_inducing,
        )

    f_train = neal_func(X_np[:stream_start_idx])
    mu_base = float(np.mean(f_train)) if len(f_train) > 0 else 0.0
    var_base = float(np.var(f_train, ddof=1)) if len(f_train) > 1 else 1.0
    var_base = max(var_base, EPS)
    const_base = 0.5 * math.log(2.0 * math.pi * var_base)

    
    abnormal_idx: List[int] = []

    mse_sum = nll_sum = msll_sum = 0.0
    n_pred = 0

    t_step_list: List[float] = []
    t_pred_list: List[float] = []
    t_detect_list: List[float] = []
    t_upd_list: List[float] = []
    t_train_list: List[float] = []

    for t in range(stream_start_idx, n_total):
        t0 = time.perf_counter()

        # ---- 1) predict ----
        tp0 = time.perf_counter()
        mu_t, sigma_t = svgp.predict(X[t])
        t_pred_list.append(time.perf_counter() - tp0)
        y_t = float(y[t].item())

        # ---- 2) detect: 95% CI ----
        td0 = time.perf_counter()
        low = mu_t - 1.96 * sigma_t
        high = mu_t + 1.96 * sigma_t
        is_abn = not (low <= y_t <= high)
        if is_abn:
            abnormal_idx.append(t)
        t_detect_list.append(time.perf_counter() - td0)

        # ---- 3) update DT + optional periodic training ----
        tu0 = time.perf_counter()

        if not is_abn:
            svgp.push(X[t], y_t)
        else:
            b = beta_iadam(mu_t, y_t, sigma_t)
            if b <= beta_max:
                svgp.push(X[t], mu_t)
            else:
                svgp.push(X[t], y_t)

        step = t - stream_start_idx + 1
        if opt_every is not None and opt_every > 0 and (step % int(opt_every) == 0) and opt_epochs > 0:
            tt0 = time.perf_counter()
            svgp.train_on_DT(
                iters=opt_epochs, lr=lr,
                reset_inducing=reset_inducing_each_train,
                init_inducing=init_inducing,
            )
            t_train_list.append(time.perf_counter() - tt0)

        t_upd_list.append(time.perf_counter() - tu0)

        # ---- 4) regression metrics (true function) ----
        y_true = float(neal_func(X_np[t]))
        err = y_true - mu_t
        var_y = max(sigma_t * sigma_t, EPS)
        nll_i = 0.5 * math.log(2.0 * math.pi * var_y) + 0.5 * (err * err) / var_y
        base_i = const_base + 0.5 * ((y_true - mu_base) ** 2) / var_base

        mse_sum += err * err
        nll_sum += nll_i
        msll_sum += (nll_i - base_i)
        n_pred += 1

        t_step_list.append(time.perf_counter() - t0)

   
    true_ab = {i for i in np.where(is_out_np)[0] if i >= stream_start_idx}
    pred_ab = set(abnormal_idx)
    tp = len(true_ab & pred_ab)
    fp = len(pred_ab - true_ab)
    fn = len(true_ab - pred_ab)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_mse = mse_sum / max(n_pred, 1)
    rmse = math.sqrt(max(avg_mse, 0.0))
    smse = avg_mse / var_base
    msll = msll_sum / max(n_pred, 1)
    avg_nll = nll_sum / max(n_pred, 1)

    ms = lambda L: 1000.0 * float(np.mean(L)) if len(L) else float("nan")

    return {
        "method": "SGP-IADAM(SVGP)",
        "seed": seed,
        "outlier_type": outlier_type,
        "outlier_ratio": float(outlier_ratio),
        "n_total": int(n_total),
        "clean_prefix": int(clean_prefix),
        "stream_start_idx": int(stream_start_idx),

        "q": int(q_eff),
        "M": int(M_eff),
        "beta_max": float(beta_max),

        "init_lengthscale": float(init_lengthscale),
        "init_variance": float(init_variance),
        "init_inducing": str(init_inducing),

        "warmup_iters": int(warmup_iters),
        "opt_every": int(opt_every) if opt_every is not None else None,
        "opt_epochs": int(opt_epochs),

        "rmse": rmse,
        "smse": smse,
        "msll": msll,
        "nll": avg_nll,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "n_pred": int(n_pred),
        "true_outliers": int(len(true_ab)),
        "detected_outliers": int(len(pred_ab)),

        "avg_step_ms": ms(t_step_list),
        "avg_pred_ms": ms(t_pred_list),
        "avg_detect_ms": ms(t_detect_list),
        "avg_upd_ms": ms(t_upd_list),
        "avg_train_ms": ms(t_train_list),

        "include_opt_in_update": bool(include_opt_in_update),
        "reset_inducing_each_train": bool(reset_inducing_each_train),
    }
