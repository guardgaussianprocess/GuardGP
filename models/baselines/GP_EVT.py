import torch
import matplotlib.pyplot as plt
import math

dtype = torch.float64
device = torch.device("cpu")

# RBF kernel with squared exponential
def rbf_kernel(X1, X2, lengthscale, variance):
    X1 = X1 / lengthscale
    X2 = X2 / lengthscale
    sqdist = torch.cdist(X1, X2, p=2).pow(2)
    return variance * torch.exp(-0.5 * sqdist)

class OnlineGP(torch.nn.Module):
    """
    Online Gaussian Process regressor using PyTorch.
    Supports batch fit, incremental Cholesky update, hyperparameter optimization,
    and removal of the last data point for a sliding window.
    """
    def __init__(self,
                 input_dim=1,
                 init_lengthscale=1.0,
                 init_variance=1.0,
                 init_noise=1e-2,
                 jitter=1e-6,
                 max_points=None):
        super().__init__()
        # Log-space parameters for stability
        self.log_lengthscale = torch.nn.Parameter(torch.log(torch.ones(input_dim, dtype=dtype) * init_lengthscale))
        self.log_variance    = torch.nn.Parameter(torch.log(torch.tensor(init_variance, dtype=dtype)))
        self.log_noise       = torch.nn.Parameter(torch.log(torch.tensor(init_noise, dtype=dtype)))
        self.jitter = jitter
        self.train_x = None
        self.train_y = None
        self.chol    = None
        self.alpha   = None
        self.is_fitted = False
        self.max_points = max_points

    def fit(self, X, y):
        """Batch fit: compute Cholesky factor and alpha coefficients."""
        self.train_x = X.to(device=device, dtype=dtype)
        self.train_y = y.to(device=device, dtype=dtype)
        n = self.train_x.size(0)
        ls    = torch.exp(self.log_lengthscale)
        var   = torch.exp(self.log_variance)
        noise = torch.exp(self.log_noise)
        K = rbf_kernel(self.train_x, self.train_x, ls, var)
        K += (noise**2 + self.jitter) * torch.eye(n, dtype=dtype, device=device)
        self.chol = torch.linalg.cholesky(K)
        #z = torch.linalg.solve(self.chol, self.train_y.unsqueeze(1))
        #self.alpha = torch.linalg.solve(self.chol.T, z).squeeze(-1)
        self.z = torch.linalg.solve(self.chol, self.train_y.unsqueeze(1))
        self.alpha = torch.linalg.solve(self.chol.T, self.z).squeeze(-1)
        self.is_fitted = True
        

    def predict(self, X_test):
        """Predict posterior mean and variance at X_test."""
        assert self.is_fitted, "Model must be fitted before predicting."
        Xs = X_test.to(device=device, dtype=dtype)
        ls    = torch.exp(self.log_lengthscale)
        var_  = torch.exp(self.log_variance)
        K_s   = rbf_kernel(self.train_x, Xs, ls, var_)
        K_ss  = rbf_kernel(Xs, Xs, ls, var_)
        mean  = K_s.T.matmul(self.alpha)
        v     = torch.linalg.solve(self.chol, K_s)
        cov   = K_ss - v.T.matmul(v)
        return mean, torch.diag(cov)
    
    def predict_y(self, X_test):
        mean_f, var_f = self.predict(X_test)  
        noise = torch.exp(self.log_noise)
        var_y = var_f + noise**2
        return mean_f, var_f,var_y.clamp_min(self.jitter)

    def update(self, x_new, y_new):
        #Incremental Cholesky update with optional sliding window.
        x = x_new.to(device=device, dtype=dtype).view(1, -1)
        y = torch.tensor([y_new], device=device, dtype=dtype)

        
        if self.is_fitted and (self.max_points is not None):
            if self.train_x.size(0) >= self.max_points:
             
                self.remove_point_schur(0)

        
        if not self.is_fitted or self.train_x is None or self.train_x.numel() == 0:
            self.fit(x, y)
            return None

        ls    = torch.exp(self.log_lengthscale)
        var_  = torch.exp(self.log_variance)
        noise = torch.exp(self.log_noise)

        # Compute cross-covariances
        k_vec = rbf_kernel(self.train_x, x, ls, var_).squeeze(1)
        k_self = rbf_kernel(x, x, ls, var_).item() + noise**2 + self.jitter

        # Solve for projection
        w = torch.linalg.solve(self.chol, k_vec.unsqueeze(1)).squeeze(1)

        # Schur complement for new diag
        schur = (k_self - torch.dot(w, w)).clamp(min=self.jitter)
        gamma = torch.sqrt(schur)

        # Expand Cholesky factor
        n = self.train_x.size(0)
        new_chol = torch.zeros((n+1, n+1), dtype=dtype, device=device)
        new_chol[:n, :n] = self.chol
        new_chol[n, :n]  = w
        new_chol[n, n]   = gamma
        self.chol = new_chol

        # Append data
        self.train_x = torch.cat([self.train_x, x], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

       
        z_last = (y_new - torch.dot(w, self.z.squeeze(1))) / gamma
        self.z  = torch.cat([self.z, z_last.view(1,1)], dim=0)

        
        u = torch.linalg.solve(self.chol[:-1, :-1].T, w.unsqueeze(1)).squeeze(1)

        alpha_last = z_last / gamma                      
        alpha_head = self.alpha - u * alpha_last          # O(n)
        self.alpha = torch.cat([alpha_head, alpha_last.view(1)], dim=0)

        return schur


    def negative_log_likelihood(self):
        """Compute negative log marginal likelihood."""
        n = self.train_x.size(0)
        ls    = torch.exp(self.log_lengthscale)
        var_  = torch.exp(self.log_variance)
        noise = torch.exp(self.log_noise)
        K = rbf_kernel(self.train_x, self.train_x, ls, var_)
        K += (noise**2 + self.jitter) * torch.eye(n, dtype=dtype, device=device)
        L = torch.linalg.cholesky(K)
        z = torch.linalg.solve(L, self.train_y.unsqueeze(1))
        alpha = torch.linalg.solve(L.T, z)
        nll = 0.5 * self.train_y.matmul(alpha.squeeze(-1))
        nll += torch.log(torch.diag(L)).sum()
        nll += 0.5 * n * torch.log(torch.tensor(2 * torch.pi, dtype=dtype))
        return nll

    def optimize(self, epochs=500, lr=0.01, verbose=True):
        """Optimize hyperparameters via gradient descent on NLL."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(1, epochs+1):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            if verbose and i % (epochs // 5) == 0:
                print(f"Epoch {i}/{epochs}, NLL: {loss.item():.4f}")
        # Refit with optimized hyperparameters
        self.fit(self.train_x, self.train_y)

    def optimize_step(self, lr=0.01):
        """Perform a single gradient descent step on hyperparameters."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = self.negative_log_likelihood()
        loss.backward()
        optimizer.step()
        self.fit(self.train_x, self.train_y)
    
       

    def _scalar_lengthscale(self):
        
        ls = torch.exp(self.log_lengthscale.detach())
        if ls.numel() == 1:
            return float(ls.item())
        else:
            return float(ls.mean().item())

    def effective_sample_size(self, x_star, h_factor: float = 2.0):
        
        assert self.is_fitted, "Model must be fitted before calling effective_sample_size."

        if x_star.dim() == 1:
            x_star = x_star.view(1, -1)
        x_star = x_star.to(device=device, dtype=dtype)

        ls_scalar = self._scalar_lengthscale()
        h = h_factor * ls_scalar
        
        h = max(h, 1e-8)

        # ||x_i - x*||^2
        diff = self.train_x - x_star  # (N, d)
        r2 = (diff ** 2).sum(dim=1)   # (N,)

        # phi_h = exp(- ||x - x*||^2 / (2 h^2))
        phi = torch.exp(-0.5 * r2 / (h * h))

        n_eff = phi.sum().item()
        
        n_eff = max(n_eff, 2.0)
        return n_eff

    def _gumbel_params_from_n(self, n_eff: float):
        
        n = float(n_eff)
        ln_n = math.log(n)
        two_ln_n = 2.0 * ln_n
        sqrt_two_ln_n = math.sqrt(two_ln_n)

        # a = (2 log n)^{-1/2}
        a = 1.0 / sqrt_two_ln_n

        # b = sqrt(2 log n) - [log log n + log(2π)] / (2 sqrt(2 log n))
        b = sqrt_two_ln_n - (math.log(ln_n) + math.log(2.0 * math.pi)) / (2.0 * sqrt_two_ln_n)

        return a, b

    def _gumbel_quantile(self, n_eff: float, p: float):
        
        a, b = self._gumbel_params_from_n(n_eff)
        
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        z_p = b - a * math.log(-math.log(p))
        return z_p



    def detect_evt(self, x_new, y_new, p=0.95, h_factor=2.0, use_var_y=True, mu=None, var=None):

        assert self.is_fitted

        # 1) GP prediction
        x = x_new.to(device=device, dtype=dtype).view(1, -1)
    

        var_clamped = var.clamp_min(self.jitter)
        std = torch.sqrt(var_clamped)
        mu_val = float(mu.item())
        std_val = float(std.item())
        y = float(y_new)

        D = abs(y - mu_val) / max(std_val, 1e-12)

        # 3) n_eff (Eq. 23)
        n_eff = self.effective_sample_size(x, h_factor=h_factor)

        # 4) z_p (Eq. 24-25 + quantile eq.)
        z_p = self._gumbel_quantile(n_eff, p)

        is_anom = D > z_p

        detail = {
            "y": y,
            "mu": mu_val,
            "std": std_val,
            "D": D,
            "z_p": z_p,
            "n_eff": n_eff
        }

        return float(D), bool(is_anom), detail
