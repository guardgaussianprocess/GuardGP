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
        """Incremental Cholesky update without full re-fit."""
        x = x_new.to(device=device, dtype=dtype).view(1, -1)
        y = torch.tensor([y_new], device=device, dtype=dtype)
        
        if not self.is_fitted:
            self.fit(x, y)
            return
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
        # Recompute alpha
        #z = torch.linalg.solve(self.chol, self.train_y.unsqueeze(1))
        #self.alpha = torch.linalg.solve(self.chol.T, z).squeeze(1)
       
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
            #if verbose and i % (epochs // 5) == 0:
                #print(f"Epoch {i}/{epochs}, NLL: {loss.item():.4f}")
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
    
   
