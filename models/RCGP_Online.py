"""Online variant of Robust and Conjugate Gaussian Process regression (RCGP).

Reference:
    Altamirano, Briol, Knoblauch. "Robust and Conjugate Gaussian Process
    Regression." ICML 2024. (arXiv:2311.00463)

RCGP replaces the Gaussian likelihood by a generalised (weighted score
matching) loss and stays fully conjugate. With a weight function w(x, y),
the posterior has the same closed form as a standard GP but with

    A     = K + sigma_n^2 * J_w,        J_w = diag( sigma_n^2 / (2 w_i^2) )
    mu(x) = m(x) + k(x)^T A^{-1} (y - m_w)
    m_w,i = m(x_i) + sigma_n^2 * d/dy_i [ log w(x_i, y_i)^2 ]
    var(x)= k(x, x) - k(x)^T A^{-1} k(x)

With the recommended IMQ weight

    w(x, y) = beta * (1 + (y - c_ctr(x))^2 / c^2)^{-1/2},   beta = sigma_n / sqrt(2),

nominal points (|y - c_ctr| << c) give J_w,ii ~ 1 so the standard GP is
recovered, while outliers get J_w,ii growing quadratically in the residual,
i.e. an effectively infinite noise variance -> soft rejection.

Online variant implemented here (single pass over the stream):
  * hyperparameters (ARD RBF lengthscales, signal variance, noise) are
    optimised once on the clean warm-up prefix via standard GP NLL
    (same protocol as all baselines in the paper), then frozen;
  * a FIFO buffer of bounded capacity C stores (x_i, y_i, w_i); the weight
    of a sample is computed once, on arrival, using the *pre-update*
    predictive mean as the centring c_ctr (option ``center='pred'``,
    default) or the constant prior mean (``center='prior'``, the exact
    setting of the paper);
  * after every insertion/eviction the Cholesky factor of A is rebuilt at
    O(C^3); with C <= 120 this costs well under a millisecond.

RCGP produces no explicit per-sample detections (it soft-downweights), so
it enters the regression comparison (RMSE / MSLL) only.
"""

from typing import Optional, Tuple

import math
import torch

from models.OnlineGP_Step import OnlineGP, rbf_kernel

DTYPE = torch.float64
DEVICE = torch.device("cpu")


class OnlineRCGP:
    def __init__(
        self,
        input_dim: int = 1,
        capacity: int = 60,
        center: str = "pred",          # 'pred' | 'prior'
        c_mode: str = "quantile",      # 'quantile' | 'sigma'
        c_quantile: float = 0.95,
        c_mult: float = 2.0,
        c_sigma_mult: float = 3.0,
        jitter: float = 1e-6,
        jw_clip: float = 1e8,
    ):
        assert center in ("pred", "prior")
        assert c_mode in ("quantile", "sigma")
        self.input_dim = input_dim
        self.capacity = int(capacity)
        self.center = center
        self.c_mode = c_mode
        self.c_quantile = float(c_quantile)
        self.c_mult = float(c_mult)
        self.c_sigma_mult = float(c_sigma_mult)
        self.jitter = float(jitter)
        self.jw_clip = float(jw_clip)

        # hyperparameters (filled by warmup_fit)
        self.lengthscale: Optional[torch.Tensor] = None
        self.variance: Optional[float] = None
        self.sigma_n: Optional[float] = None
        self.m0: float = 0.0           # constant prior mean
        self.c: Optional[float] = None  # IMQ soft-threshold scale
        self.beta: Optional[float] = None

        # buffer state
        self.X = torch.empty((0, input_dim), dtype=DTYPE, device=DEVICE)
        self.y = torch.empty((0,), dtype=DTYPE, device=DEVICE)
        self.ctr = torch.empty((0,), dtype=DTYPE, device=DEVICE)  # centring per point
        self.chol: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def warmup_fit(
        self,
        X_warm: torch.Tensor,
        y_warm: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.05,
        init_lengthscale: float = 1.0,
        init_variance: float = 1.0,
        init_noise: float = 0.2,
    ) -> None:
        """Optimise GP hyperparameters on the (clean) warm-up prefix with a
        standard GP marginal likelihood, then initialise the RCGP buffer
        with the warm-up data (unit weights: warm-up is trusted, as for all
        baselines)."""
        X_warm = X_warm.to(DEVICE, DTYPE).view(-1, self.input_dim)
        y_warm = y_warm.to(DEVICE, DTYPE).view(-1)

        gp = OnlineGP(
            input_dim=self.input_dim,
            init_lengthscale=init_lengthscale,
            init_variance=init_variance,
            init_noise=init_noise,
        )
        gp.fit(X_warm, y_warm)
        gp.optimize(epochs=epochs, lr=lr, verbose=False)

        self.lengthscale = torch.exp(gp.log_lengthscale.detach()).clone()
        self.variance = float(torch.exp(gp.log_variance.detach()).item())
        self.sigma_n = float(torch.exp(gp.log_noise.detach()).item())
        self.beta = self.sigma_n / math.sqrt(2.0)
        self.m0 = float(y_warm.mean().item())

        # IMQ scale c from warm-up residuals of the fitted GP
        with torch.no_grad():
            mu_w, _, _ = gp.predict_y(X_warm)
        res = (y_warm - mu_w).abs()
        if self.c_mode == "quantile":
            base = float(torch.quantile(res, self.c_quantile).item())
            self.c = max(self.c_mult * base, 1e-6)
        else:
            self.c = max(self.c_sigma_mult * self.sigma_n, 1e-6)

        # seed the buffer with the (most recent) warm-up tail
        keep = min(self.capacity, X_warm.size(0))
        self.X = X_warm[-keep:].clone()
        self.y = y_warm[-keep:].clone()
        # warm-up samples are centred at the prior mean; their residuals are
        # nominal so the weights are ~beta and J_w ~ I either way
        self.ctr = torch.full((keep,), self.m0, dtype=DTYPE, device=DEVICE)
        self._refit()

    # ------------------------------------------------------------------ #
    def _weights(self) -> torch.Tensor:
        r = self.y - self.ctr
        return self.beta * torch.rsqrt(1.0 + (r / self.c) ** 2)

    def _refit(self) -> None:
        """Rebuild Cholesky of A = K + sigma_n^2 J_w and alpha = A^{-1}(y - m_w)."""
        n = self.X.size(0)
        if n == 0:
            self.chol = None
            self.alpha = None
            return
        s2 = self.sigma_n ** 2
        w = self._weights()
        jw = (s2 / (2.0 * w ** 2)).clamp(max=self.jw_clip)

        K = rbf_kernel(self.X, self.X, self.lengthscale, self.variance)
        A = K + torch.diag(s2 * jw) + self.jitter * torch.eye(n, dtype=DTYPE, device=DEVICE)
        self.chol = torch.linalg.cholesky(A)

        # m_w,i = m0 + s2 * d/dy log w^2 = m0 - 2 s2 (y - ctr) / (c^2 + (y - ctr)^2)
        r = self.y - self.ctr
        m_w = self.m0 - 2.0 * s2 * r / (self.c ** 2 + r ** 2)
        ytil = (self.y - m_w).unsqueeze(1)
        z = torch.linalg.solve_triangular(self.chol, ytil, upper=False)
        self.alpha = torch.linalg.solve_triangular(self.chol.T, z, upper=True).squeeze(1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[float, float, float]:
        """Return (mean, latent var, predictive var incl. noise) at a single x."""
        x = x.to(DEVICE, DTYPE).view(1, -1)
        if self.chol is None:
            v = self.variance if self.variance is not None else 1.0
            s2 = self.sigma_n ** 2 if self.sigma_n is not None else 1.0
            return self.m0, v, v + s2
        k_s = rbf_kernel(self.X, x, self.lengthscale, self.variance)     # (n,1)
        k_ss = float(rbf_kernel(x, x, self.lengthscale, self.variance).item())
        mean = self.m0 + float((k_s.squeeze(1) * self.alpha).sum().item())
        v = torch.linalg.solve_triangular(self.chol, k_s, upper=False)
        var = max(k_ss - float((v ** 2).sum().item()), self.jitter)
        return mean, var, var + self.sigma_n ** 2

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, x: torch.Tensor, y_val: float, mu_pred: Optional[float] = None) -> None:
        """Insert one observation. ``mu_pred`` is the pre-update predictive
        mean at x (used as the IMQ centring when center='pred')."""
        x = x.to(DEVICE, DTYPE).view(1, -1)
        if self.center == "pred" and mu_pred is not None:
            ctr = float(mu_pred)
        else:
            ctr = self.m0
        self.X = torch.cat([self.X, x], dim=0)
        self.y = torch.cat([self.y, torch.tensor([float(y_val)], dtype=DTYPE, device=DEVICE)])
        self.ctr = torch.cat([self.ctr, torch.tensor([ctr], dtype=DTYPE, device=DEVICE)])
        if self.X.size(0) > self.capacity:   # FIFO eviction
            self.X = self.X[1:]
            self.y = self.y[1:]
            self.ctr = self.ctr[1:]
        self._refit()
