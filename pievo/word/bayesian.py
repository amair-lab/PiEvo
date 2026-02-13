import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import minimize

class GaussianProcessModel:
    """
    Minimalist Academic Implementation: Exact Gaussian Process.
    Models y = f(φ(h,P)) + ε, where f ~ GP(0, k(·,·)) and ε ~ N(0,σ²).
    Uses a standard Isotropic RBF kernel.
    """

    def __init__(
            self,
            principle_id: str,
            feature_dim: int = 2,
            length_scale: float = 1.0,
            signal_variance: float = 1.0,
            noise_variance: float = 0.1,
            log_dir: Optional[str] = None,
    ):
        self.principle_id = principle_id
        self.feature_dim = feature_dim
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance

        # Data storage
        self.X = []  # Input features
        self.y = []  # Outputs
        self.n_observations = 0

        # Cached computations
        self._K_inv = None
        self._alpha = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Isotropic RBF kernel k(x,x') = σ²_f exp(-||x-x'||² / (2 * ℓ²))"""
        if X1.ndim == 1: X1 = X1.reshape(1, -1)
        if X2.ndim == 1: X2 = X2.reshape(1, -1)

        sq_dists = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return self.signal_variance * np.exp(-0.5 * sq_dists / (self.length_scale ** 2))

    def update(self, phi: np.ndarray, y: float):
        """Update GP with new observation. Recomputes posterior exactly."""
        phi = phi.flatten()
        self.X.append(phi)
        self.y.append(y)
        self.n_observations += 1
        
        # Reset cache for retraining
        self._K_inv = None
        self._alpha = None

    def _update_cache(self):
        """Compute K^-1 and alpha for exact inference."""
        if self._K_inv is not None:
            return

        X_train = np.array(self.X)
        y_train = np.array(self.y)
        
        K = self._rbf_kernel(X_train, X_train)
        K += (self.noise_variance + 1e-6) * np.eye(len(X_train)) # Add jitter

        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ y_train

    def predict(self, phi: np.ndarray) -> Tuple[float, float]:
        """Exact GP prediction for mean and variance."""
        phi = phi.flatten().reshape(1, -1)
        
        if self.n_observations == 0:
            return 0.0, self.signal_variance + self.noise_variance

        self._update_cache()
        X_train = np.array(self.X)
        
        k_star = self._rbf_kernel(X_train, phi).flatten()
        mean = float(k_star.T @ self._alpha)
        
        variance = (
            self.signal_variance + self.noise_variance - k_star.T @ self._K_inv @ k_star
        )
        return mean, max(variance, 1e-12)

    def optimize_hyperparameters(self):
        """Simple MLE-based hyperparameter optimization for [ℓ, σ_f, σ_n]."""
        if self.n_observations < 3:
            return

        def negative_log_likelihood(params):
            l, sf, sn = np.exp(params)
            # Temporarily set params
            old_params = (self.length_scale, self.signal_variance, self.noise_variance)
            self.length_scale, self.signal_variance, self.noise_variance = l, sf, sn
            
            X_train = np.array(self.X)
            y_train = np.array(self.y)
            K = self._rbf_kernel(X_train, X_train) + (sn + 1e-6) * np.eye(len(X_train))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * (y_train.T @ alpha + log_det + len(y_train) * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                nll = 1e12
                
            # Restore
            self.length_scale, self.signal_variance, self.noise_variance = old_params
            return nll

        initial_params = np.log([self.length_scale, self.signal_variance, self.noise_variance])
        res = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B')
        
        if res.success:
            self.length_scale, self.signal_variance, self.noise_variance = np.exp(res.x)
            self._K_inv = None # Forces recompute