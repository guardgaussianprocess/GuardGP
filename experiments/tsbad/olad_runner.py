"""OLAD baseline: Xu, Kersting & von Ritter, "Stochastic Online Anomaly
Analysis for Streaming Time Series", IJCAI 2017.

Model: the stream y_t = f(t) follows a Student-t process with SE kernel and
integrated noise, i.e. y ~ MVT_n(nu, 0, K + sigma_n^2 I). One-step-ahead
predictive distribution (paper Eq. 3):

    p(f_* | y) = UVT(nu_*, m_*, s_*^2),
    nu_* = nu + n,  m_* = k_*^T A^-1 y,
    beta = y^T A^-1 y,  s^2 = k_** - k_*^T A^-1 k_*,
    s_*^2 = (nu + beta - 2) / (nu_* - 2) * s^2,   A = K + sigma_n^2 I.

Hyperparameters (rho, ell, sigma_n, nu) are fitted on the warm-up prefix by
minimizing the joint Student-t NLL (paper Eq. 8) — this is OLAD-1; with
sgd_lr > 0 they are additionally adapted online by SGD on the negative log
predictive likelihood of each new observation — OLAD-2. We use autograd in
place of the paper's manual gradients.

Streaming: a sliding window of the most recent `window` observations (the
paper discards distant history since the kernel decays). All observations
enter the window (robustness comes from the heavy-tailed likelihood, as in
the paper). Score = |y - m_*| / s_*; alarm = observation outside the
P = 99.99% predictive interval (the paper's setting).
"""

import math
import os
import sys

import numpy as np
import torch
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DTYPE = torch.float64


def _mvt_nll(y, K_noisy, nu):
    """Joint Student-t NLL of y under MVT(nu, 0, K_noisy) (paper Eq. 8),
    constants in n dropped."""
    n = y.shape[0]
    L = torch.linalg.cholesky(K_noisy)
    a = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
    beta = torch.dot(y, a)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    return (logdet + n * torch.log(nu - 2.0)
            + 2.0 * (torch.lgamma(nu / 2.0) - torch.lgamma((nu + n) / 2.0))
            + (nu + n) * torch.log1p(beta / (nu - 2.0)))


def _se_kernel(t1, t2, log_rho, log_ell):
    d2 = (t1.view(-1, 1) - t2.view(1, -1)) ** 2
    return torch.exp(2.0 * log_rho) * torch.exp(-0.5 * d2 / torch.exp(2.0 * log_ell))


class OLAD:
    def __init__(self, window=200, p_alarm=0.9999, sgd_lr=0.0, jitter=1e-8):
        self.window = window
        self.p_alarm = p_alarm
        self.sgd_lr = sgd_lr
        self.jitter = jitter
        self.params = None
        self.ts, self.ys = [], []

    # ---- hyperparameters ----
    def _unpack(self):
        lr_, le_, ln_, lnu_ = self.params
        return lr_, le_, torch.exp(2.0 * ln_), 2.0 + torch.exp(lnu_)

    def fit_warmup(self, t, y, epochs=200, lr=0.05):
        t = torch.as_tensor(t, dtype=DTYPE)
        y = torch.as_tensor(y, dtype=DTYPE)
        self.params = [torch.tensor(v, dtype=DTYPE, requires_grad=True)
                       for v in (0.0, 0.0, math.log(0.3), math.log(3.0))]
        opt = torch.optim.Adam(self.params, lr=lr)
        n = len(y)
        for _ in range(epochs):
            opt.zero_grad()
            lr_, le_, sn2, nu = self._unpack()
            K = _se_kernel(t, t, lr_, le_) + (sn2 + self.jitter) * torch.eye(n, dtype=DTYPE)
            loss = _mvt_nll(y, K, nu)
            loss.backward()
            opt.step()
        self.ts = list(t[-self.window:].detach())
        self.ys = list(y[-self.window:].detach())

    # ---- one step ----
    def step(self, t_new, y_new):
        """Predict at t_new, score y_new, optionally SGD-adapt, absorb."""
        t = torch.stack(self.ts)
        y = torch.stack(self.ys)
        n = len(y)
        do_sgd = self.sgd_lr > 0
        ctx = torch.enable_grad() if do_sgd else torch.no_grad()
        with ctx:
            lr_, le_, sn2, nu = self._unpack()
            K = _se_kernel(t, t, lr_, le_) + (sn2 + self.jitter) * torch.eye(n, dtype=DTYPE)
            L = torch.linalg.cholesky(K)
            ks = _se_kernel(t, torch.tensor([t_new], dtype=DTYPE), lr_, le_).squeeze(1)
            kss = torch.exp(2.0 * lr_)
            a = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
            b = torch.cholesky_solve(ks.unsqueeze(1), L).squeeze(1)
            beta = torch.dot(y, a)
            m_star = torch.dot(ks, a)
            s2 = torch.clamp(kss + sn2 - torch.dot(ks, b), min=self.jitter)
            nu_star = nu + n
            s_star2 = (nu + beta - 2.0) / (nu_star - 2.0) * s2

            if do_sgd:
                # negative log predictive likelihood of y_new (univariate t)
                z2 = (torch.tensor(y_new, dtype=DTYPE) - m_star) ** 2 / s_star2
                nll = (0.5 * torch.log(s_star2)
                       + torch.lgamma(nu_star / 2.0)
                       - torch.lgamma((nu_star + 1.0) / 2.0)
                       + 0.5 * torch.log(nu_star - 2.0)
                       + (nu_star + 1.0) / 2.0 * torch.log1p(z2 / (nu_star - 2.0)))
                grads = torch.autograd.grad(nll, self.params)
                with torch.no_grad():
                    for p, g in zip(self.params, grads):
                        if torch.isfinite(g):
                            p -= self.sgd_lr * g

        m = float(m_star.item())
        s = math.sqrt(float(s_star2.item()))
        nu_s = float(nu_star.item())
        score = abs(y_new - m) / max(s, 1e-12)
        # standardized alarm quantile of the unit-variance t_(nu*)
        q = tdist.ppf(0.5 + self.p_alarm / 2.0, df=nu_s) * math.sqrt(
            max((nu_s - 2.0) / nu_s, 1e-12))
        alarm = score > q

        self.ts.append(torch.tensor(t_new, dtype=DTYPE))
        self.ys.append(torch.tensor(float(y_new), dtype=DTYPE))
        if len(self.ys) > self.window:
            self.ts.pop(0)
            self.ys.pop(0)
        return score, alarm, m, s, nu_s


def run_olad_series(y: np.ndarray, warmup_n: int, *, window=200,
                    sgd_lr=0.0, p_alarm=0.9999, seed=0):
    """Returns (scores, alarms). sgd_lr=0 -> OLAD-1, >0 -> OLAD-2."""
    torch.manual_seed(seed)
    y = np.asarray(y, float)
    n = len(y)
    x = np.arange(n, dtype=float)
    x_mean, x_std = x[:warmup_n].mean(), x[:warmup_n].std() + 1e-6
    y_mean, y_std = y[:warmup_n].mean(), y[:warmup_n].std() + 1e-9
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std

    det = OLAD(window=window, p_alarm=p_alarm, sgd_lr=sgd_lr)
    det.fit_warmup(xs[:warmup_n], ys[:warmup_n])
    scores = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)
    for t in range(warmup_n, n):
        scores[t], alarms[t], _, _, _ = det.step(float(xs[t]), float(ys[t]))
    return scores, alarms


def run_olad_regression(X: np.ndarray, y_obs: np.ndarray, warmup_n: int, *,
                        window=100, sgd_lr=0.0, seed=0):
    """Streaming one-step-ahead prediction for regression benchmarks.
    X: (N,) inputs; returns per-step predictive mean, std (unit-variance
    scale) and dof of the Student-t predictive, in the y units used here
    (caller standardizes/destandardizes)."""
    torch.manual_seed(seed)
    X = np.asarray(X, float).ravel()
    y_obs = np.asarray(y_obs, float).ravel()
    n = len(y_obs)
    det = OLAD(window=window, sgd_lr=sgd_lr)
    det.fit_warmup(X[:warmup_n], y_obs[:warmup_n])
    mu = np.zeros(n)
    sd = np.zeros(n)
    dof = np.zeros(n)
    for t in range(warmup_n, n):
        _, _, mu[t], sd[t], dof[t] = det.step(float(X[t]), float(y_obs[t]))
    return mu, sd, dof
