from typing import Any
import math
from collections import deque

import torch

from models.OnlineGP_Step import OnlineGP
from models.evt_spot import surprise_from_gaussian
EPS = 1e-12

def clone_hypers(src_gp, dst_gp):
    
    with torch.no_grad():
        if hasattr(src_gp, 'log_lengthscale') and hasattr(dst_gp, 'log_lengthscale'):
            dst_gp.log_lengthscale.data.copy_(src_gp.log_lengthscale.data)
        if hasattr(src_gp, 'log_variance') and hasattr(dst_gp, 'log_variance'):
            dst_gp.log_variance.data.copy_(src_gp.log_variance.data)
        if hasattr(src_gp, 'log_noise') and hasattr(dst_gp, 'log_noise'):
            dst_gp.log_noise.data.copy_(src_gp.log_noise.data)

def short_tune(gp, epochs=0, lr=0.02):
    
    try:
        if epochs > 0:
            gp.optimize(epochs=epochs, lr=lr)
    except Exception as e:
        print("[warn] short_tune failed:", e)


def greedy_farthest_points(X: torch.Tensor, k: int) -> torch.Tensor:
   
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
    X_core: torch.Tensor, y_core: torch.Tensor,  
    recent_x: deque, recent_y: deque,            
    K_init: int,
    cover_frac: float = 0.6,
    window_mult: int = 5,
    device=None,
    dtype=torch.float64,
):
   
    device = device or X_core.device
    K_init = int(K_init)

    
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

   
    if idx.numel() < K_init:
        
        for i in range(Nc + Nr - 1, -1, -1):
            if i not in selected:
                idx = torch.cat([idx, torch.tensor([i], device=device, dtype=torch.long)])
                selected.add(i)
            if idx.numel() >= K_init:
                break

    return X_cand[idx[:K_init]], y_cand[idx[:K_init]]

# ========= GPShard / ShardManager =========
class GPShard:
    def __init__(self, input_dim=1, device="cpu", dtype=torch.float64):
        self.gp = OnlineGP(
            input_dim=input_dim,
            init_lengthscale=1,
            init_variance=1,
            init_noise=0.2
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
    assert n_total >= critic_seed_min, (
        f"Not enough samples in critic_recent "
        f"(required >= {critic_seed_min}, got {n_total})"
    )

    n_take = min(k_critic_seed_recent, n_total)
    Xr = torch.stack(list(critic_recent_x), 0).to(device, dtype)[-n_take:]
    yr = torch.tensor(list(critic_recent_y),
                      device=device, dtype=dtype)[-n_take:]
    return Xr, yr
