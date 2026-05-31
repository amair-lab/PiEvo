import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Sequence, Tuple

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from scipy.optimize import minimize

BASE_SIZE = 14  # Base font size

plt.rcParams.update(
    {
        "font.size": BASE_SIZE,  # Default font size for text
        "axes.titlesize": BASE_SIZE + 6,  # Subplot title size (e.g., 28)
        "figure.titlesize": BASE_SIZE
        + 8,  # Main figure title (subtitle) size (e.g., 30)
        "axes.labelsize": BASE_SIZE + 2,  # X and Y axis label size (e.g., 24)
        "xtick.labelsize": BASE_SIZE - 2,  # X tick label size (e.g., 20)
        "ytick.labelsize": BASE_SIZE - 2,  # Y tick label size (e.g., 20)
        "legend.fontsize": BASE_SIZE,  # Legend font size
        "font.family": "serif",  # Use 'serif' style
    }
)

# Try to import sklearn for PCA, but don't fail if not installed
# It will be checked for again inside the plotting function
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None  # Flag that PCA is not available

logger = logging.getLogger(__name__)


class GaussianProcessModel:
    """
    Gaussian Process model for nonlinear relationship modeling between principles and outcomes.
    Implements y = f(φ(h,P)) + ε, where f ~ GP(0, k(·,·)) and ε ~ N(0,σ²)

    Supports incremental learning with online GP updates for computational efficiency.

    Includes methods to generate and save diagnostic plotting data as JSON,
    and to render academic-style reports from either the live model or the saved JSON data.
    """

    def __init__(
        self,
        principle_id: str,
        feature_dim: int = 4,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        max_inducing_points: int = 24,
        anisotropic: bool = True,
        optimize_hyperparams: bool = True,
        optimization_interval: int = 5,
        log_dir: Optional[str] = None,
        is_rbf: bool = True,
    ):
        self.feature_dim = feature_dim
        self.anisotropic = anisotropic
        self.optimize_hyperparams = optimize_hyperparams
        self.optimization_interval = optimization_interval
        self.is_rbf = is_rbf

        # Ensure log_dir is set, default to current directory if None
        base_log_dir = log_dir if log_dir is not None else "."
        self.log_dir = os.path.join(base_log_dir, "gaussian_process")
        os.makedirs(self.log_dir, exist_ok=True)
        self.principle_id = principle_id

        # Initialize hyperparameters - use anisotropic if enabled
        if anisotropic:
            self.length_scale = (
                np.full(feature_dim, length_scale)
                if isinstance(length_scale, (int, float))
                else np.array(length_scale)
            )
        else:
            self.length_scale = float(length_scale)

        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.max_inducing_points = max_inducing_points

        # Training data storage
        self.X = []  # Input features
        self.y = []  # Outputs
        self.n_observations = 0

        # Inducing points for sparse GP (computational efficiency)
        self.inducing_X = None
        self.inducing_y = None

        # Cached computations for incremental updates
        self._K_inv = None
        self._alpha = None
        self._last_update_size = 0

        # Optimization tracking
        self._last_optimization_size = 0

        # Store history of hyperparameter optimization
        self.hyperparam_history = []
        self._log_hyperparameters(0)

    def _log_hyperparameters(self, n_obs: int):
        """Helper to log the current hyperparameters to the history list."""
        self.hyperparam_history.append(
            {
                "n_observations": n_obs,
                "signal_variance": self.signal_variance,
                "noise_variance": self.noise_variance,
                "length_scale_mean": np.mean(self.length_scale)
                if self.anisotropic
                else self.length_scale,
                "length_scale_std": np.std(self.length_scale)
                if self.anisotropic
                else 0.0,
            }
        )

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel k(x,x') = σ²_f exp(-||x-x'||²_M/(2)) where M is the metric tensor"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        # Check and handle dimension mismatch
        actual_dim = X1.shape[1]

        if self.anisotropic:
            expected_dim = (
                self.length_scale.shape[0]
                if hasattr(self.length_scale, "shape")
                else len(self.length_scale)
            )
            if expected_dim != actual_dim:
                # Adjust length_scale to match actual feature dimension
                if hasattr(self.length_scale, "shape"):
                    old_length = len(self.length_scale)
                else:
                    old_length = (
                        len([self.length_scale])
                        if np.isscalar(self.length_scale)
                        else len(self.length_scale)
                    )

                # Create new length scale that matches the actual dimension
                if actual_dim > old_length:
                    # Pad with the last value or extend with default value
                    if np.isscalar(self.length_scale):
                        self.length_scale = np.full(actual_dim, self.length_scale)
                    else:
                        new_length_scale = np.full(
                            actual_dim,
                            self.length_scale[-1]
                            if len(self.length_scale) > 0
                            else 1.0,
                        )
                        new_length_scale[: self.length_scale.shape[0]] = (
                            self.length_scale
                        )
                        self.length_scale = new_length_scale
                else:
                    # Truncate to actual dimension
                    if np.isscalar(self.length_scale):
                        # If it was a scalar, convert to array first then truncate
                        self.length_scale = np.full(actual_dim, self.length_scale)
                    else:
                        self.length_scale = self.length_scale[:actual_dim]

                # Update feature_dim to match
                self.feature_dim = actual_dim
                # print(f"GP Kernel: Updated length_scale to dimension {actual_dim}.")
        else:
            # For isotropic case, we just need to ensure self.length_scale is a scalar
            if hasattr(self.length_scale, "__len__") and len(self.length_scale) > 1:
                # Use the first value or average if it's unexpectedly an array
                if len(self.length_scale) > 0:
                    self.length_scale = float(self.length_scale[0])

        if self.is_rbf:
            if self.anisotropic:
                # Anisotropic kernel: different length scale per dimension
                # Scale features by inverse length scales
                X1_scaled = X1 / self.length_scale
                X2_scaled = X2 / self.length_scale

                # Compute scaled squared distances
                sq_dists = (
                    np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                    + np.sum(X2_scaled**2, axis=1)
                    - 2 * np.dot(X1_scaled, X2_scaled.T)
                )
            else:
                # Isotropic kernel: single length scale
                sq_dists = (
                    np.sum(X1**2, axis=1).reshape(-1, 1)
                    + np.sum(X2**2, axis=1)
                    - 2 * np.dot(X1, X2.T)
                )
                sq_dists = sq_dists / (self.length_scale**2)

            return self.signal_variance * np.exp(-0.5 * sq_dists)
        else:
            # Additive Kernel over dimensions
            K = np.zeros((X1.shape[0], X2.shape[0]))

            if self.anisotropic:
                X1_scaled = X1 / self.length_scale
                X2_scaled = X2 / self.length_scale
            else:
                X1_scaled = X1 / self.length_scale
                X2_scaled = X2 / self.length_scale

            for d in range(actual_dim):
                x1_d = X1_scaled[:, d].reshape(-1, 1)
                x2_d = X2_scaled[:, d].reshape(1, -1)

                sq_dists_d = (x1_d**2) + (x2_d**2) - 2 * (x1_d * x2_d)
                K += np.exp(-0.5 * sq_dists_d)

            # Average over dimensions to keep signal_variance scale stable
            return self.signal_variance * (K / actual_dim)

    def _select_inducing_points(self):
        """Select inducing points using greedy variance maximization for sparse GP"""
        # print(f"GP Inducing Points: Selecting from {len(self.X)} total observations (max_inducing_points={self.max_inducing_points})")

        if len(self.X) <= self.max_inducing_points:
            self.inducing_X = np.array(self.X)
            self.inducing_y = np.array(self.y)
            # print(f"GP Inducing Points: Using all {len(self.X)} observations as inducing points")
            return

        # Start with random point
        X_array = np.array(self.X)
        y_array = np.array(self.y)

        initial_idx = np.random.randint(len(X_array))
        inducing_indices = [initial_idx]
        # print(f"GP Inducing Points: Started with random point {initial_idx}")

        # Greedy selection based on variance
        for i in range(self.max_inducing_points - 1):
            remaining_indices = set(range(len(X_array))) - set(inducing_indices)
            if not remaining_indices:
                break

            best_idx = None
            best_var = -1

            # For efficiency, only evaluate every 10th remaining point if more than 100 remain
            remaining_list = list(remaining_indices)
            if len(remaining_list) > 100:
                # Sample every n-th point to speed up computation
                step = max(1, len(remaining_list) // 50)  # Evaluate ~50 points max
                evaluation_indices = remaining_list[::step]
            else:
                evaluation_indices = remaining_list

            for idx in evaluation_indices:
                current_inducing = X_array[inducing_indices]
                test_point = X_array[idx].reshape(1, -1)

                # Compute variance at this point given current inducing points
                k_star = self._rbf_kernel(current_inducing, test_point).flatten()
                try:
                    K_inv = np.linalg.inv(
                        self._rbf_kernel(current_inducing, current_inducing)
                        + self.noise_variance * np.eye(len(inducing_indices))
                    )

                    variance = self.signal_variance - k_star.T @ K_inv @ k_star
                    if variance > best_var:
                        best_var = variance
                        best_idx = idx
                except np.linalg.LinAlgError:
                    # print(f"GP Inducing Points: LinAlgError for point {idx}, skipping")
                    continue

            if best_idx is not None:
                inducing_indices.append(best_idx)
                if i % 5 == 0:  # Print progress every 5 steps
                    print(
                        f"GP Inducing Points: Selected point {best_idx}, current variance={best_var:.4f} ({i + 1}/{self.max_inducing_points})"
                    )

        self.inducing_X = X_array[inducing_indices]
        self.inducing_y = y_array[inducing_indices]
        # print(f"GP Inducing Points: Selected {len(inducing_indices)} inducing points with max variance={best_var:.4f}")

    def update(self, phi: np.ndarray, y: float):
        """Incrementally update GP with new observation"""
        phi = phi.reshape(-1) if phi.ndim == 1 else phi.flatten()

        self.X.append(phi.copy())
        self.y.append(y)
        self.n_observations += 1

        # Update inducing points periodically for efficiency
        if self.n_observations % 10 == 0 or len(self.X) <= self.max_inducing_points:
            self._select_inducing_points()
            self._invalidate_cache()

        # Optimize hyperparameters periodically
        if (
            self.optimize_hyperparams
            and self.n_observations > 5  # Need minimum data
            and (self.n_observations - self._last_optimization_size)
            >= self.optimization_interval
        ):
            self._optimize_hyperparameters()
            self._last_optimization_size = self.n_observations

    def _invalidate_cache(self):
        """Invalidate cached computations when inducing points change"""
        self._K_inv = None
        self._alpha = None
        self._last_update_size = 0

    def _update_cache(self):
        """Update cached computations for efficient prediction"""
        cache_needed_update = (
            self.inducing_X is None
            or len(self.inducing_X) == 0
            or self._K_inv is None
            or len(self.inducing_X) != self._last_update_size
        )

        if cache_needed_update:
            # print(f"GP Cache Update: Updating cache with {len(self.inducing_X) if self.inducing_X is not None else 0} inducing points")

            if self.inducing_X is not None and len(self.inducing_X) > 0:
                K = self._rbf_kernel(self.inducing_X, self.inducing_X)
                K += self.noise_variance * np.eye(len(self.inducing_X))

                try:
                    self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
                    self._alpha = self._K_inv @ self.inducing_y
                    self._last_update_size = len(self.inducing_X)
                    # print(f"GP Cache Update: Successfully computed K_inv (size: {K.shape[0]}x{K.shape[1]})")
                except np.linalg.LinAlgError:
                    # Add jitter for numerical stability
                    jitter = 1e-4
                    self._K_inv = np.linalg.inv(K + jitter * np.eye(len(K)))
                    self._alpha = self._K_inv @ self.inducing_y
                    self._last_update_size = len(self.inducing_X)
                    # print(f"GP Cache Update: Used jitter for numerical stability (jitter={jitter})")

    def predict(self, phi: np.ndarray) -> Tuple[float, float]:
        """Predict mean and variance using GP posterior"""
        phi = phi.reshape(-1) if phi.ndim == 1 else phi.flatten()

        if self.n_observations == 0:
            # Prior prediction
            return 0.0, self.signal_variance + self.noise_variance

        if self.inducing_X is None or len(self.inducing_X) == 0:
            return 0.0, self.signal_variance + self.noise_variance

        cache_was_valid = (
            self._K_inv is not None and len(self.inducing_X) == self._last_update_size
        )
        self._update_cache()
        cache_updated = not cache_was_valid and self._K_inv is not None

        # Compute kernel vector between test point and inducing points
        k_star = self._rbf_kernel(self.inducing_X, phi.reshape(1, -1)).flatten()

        # GP posterior mean
        mean = float(k_star.T @ self._alpha)

        # GP posterior variance
        variance = (
            self.signal_variance + self.noise_variance - k_star.T @ self._K_inv @ k_star
        )
        variance = max(variance, 1e-12)  # Ensure positive

        return mean, variance

    def sample_prediction(self, phi: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from GP posterior predictive distribution"""
        phi = phi.reshape(-1) if phi.ndim == 1 else phi.flatten()

        mean, variance = self.predict(phi)
        std = np.sqrt(variance)

        samples = np.random.normal(mean, std, size=n_samples)

        return samples

    def _log_marginal_likelihood(self, params: np.ndarray) -> float:
        """Compute log marginal likelihood for hyperparameter optimization"""
        # Store original hyperparameters
        old_length_scale = self.length_scale
        old_signal_var = self.signal_variance
        old_noise_var = self.noise_variance

        try:
            # Unpack hyperparameters
            if self.anisotropic:
                n_length_scales = self.feature_dim
                length_scales = np.exp(
                    params[:n_length_scales]
                )  # Use log-space for positivity
                signal_variance = np.exp(params[n_length_scales])
                noise_variance = np.exp(params[n_length_scales + 1])
            else:
                length_scale = np.exp(params[0])
                signal_variance = np.exp(params[1])
                noise_variance = np.exp(params[2])
                length_scales = length_scale

            # Temporarily set hyperparameters
            self.length_scale = length_scales
            self.signal_variance = signal_variance
            self.noise_variance = noise_variance

            # Use inducing points if available, otherwise full dataset
            if self.inducing_X is not None and len(self.inducing_X) > 0:
                X = self.inducing_X
                y = self.inducing_y
            else:
                if not self.X:
                    return -1e6  # No data available
                X = np.array(self.X)
                y = np.array(self.y)

            if len(X) == 0:
                return -1e6

            # Compute covariance matrix
            K = self._rbf_kernel(X, X) + noise_variance * np.eye(len(X))

            # Add jitter for numerical stability
            jitter = 1e-6
            K += jitter * np.eye(len(K))

            # Compute log marginal likelihood
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, y)

                # Log marginal likelihood = -0.5 * (y^T K^-1 y + log|K| + n*log(2π))
                log_likelihood = -0.5 * (
                    np.dot(alpha, alpha)
                    + 2 * np.sum(np.log(np.diag(L)))
                    + len(y) * np.log(2 * np.pi)
                )

            except np.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                try:
                    sign, log_det = np.linalg.slogdet(K)
                    if sign <= 0:
                        return -1e6
                    K_inv = np.linalg.inv(K)
                    log_likelihood = -0.5 * (
                        np.dot(y, np.dot(K_inv, y))
                        + log_det
                        + len(y) * np.log(2 * np.pi)
                    )
                except np.linalg.LinAlgError:
                    return -1e6

            # Restore original hyperparameters
            self.length_scale = old_length_scale
            self.signal_variance = old_signal_var
            self.noise_variance = old_noise_var

            return log_likelihood

        except Exception as e:
            # Restore hyperparameters on error
            try:
                self.length_scale = old_length_scale
                self.signal_variance = old_signal_var
                self.noise_variance = old_noise_var
            except:
                pass
            return -1e6

    def _optimize_hyperparameters(self):
        if not self.X or len(self.X) < 3:
            return  # Need minimum data for optimization

        self._log_hyperparameters(self.n_observations)

        try:
            # Set up initial parameters (in log space for positivity constraints)
            if self.anisotropic:
                # Initial parameters: [log(length_scales), log(signal_var), log(noise_var)]
                initial_params = np.concatenate(
                    [
                        np.log(self.length_scale),  # length scales for each dimension
                        [np.log(self.signal_variance), np.log(self.noise_variance)],
                    ]
                )
                bounds = [(np.log(1e-3), np.log(10.0))] * self.feature_dim + [
                    (np.log(1e-3), np.log(10.0)),
                    (np.log(1e-4), np.log(0.1)),
                ]  # TODO.
            else:
                # Initial parameters: [log(length_scale), log(signal_var), log(noise_var)]
                initial_params = np.array(
                    [
                        np.log(self.length_scale),
                        np.log(self.signal_variance),
                        np.log(self.noise_variance),
                    ]
                )
                bounds = [
                    (np.log(1e-3), np.log(10.0)),
                    (np.log(1e-3), np.log(10.0)),
                    (np.log(1e-4), np.log(0.1)),
                ]  # TODO.

            # Define objective (negative log marginal likelihood)
            def objective(params):
                ll = self._log_marginal_likelihood(params)
                return -ll  # Negative for minimization

            # Optimize using L-BFGS-B
            result = minimize(
                objective,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100, "disp": False},
            )

            if not result.success:
                logger.warning(
                    f"GP {self.principle_id} optimizer FAILED: {result.message}"
                )
            else:
                logger.info(
                    f"GP {self.principle_id} optimizer SUCCESS. New noise: {np.exp(result.x[-1]):.4f}"
                )

            if result.success:
                # Update hyperparameters with optimized values
                if self.anisotropic:
                    n_length_scales = self.feature_dim
                    optimized_length_scales = np.exp(result.x[:n_length_scales])
                    optimized_signal_var = np.exp(result.x[n_length_scales])
                    optimized_noise_var = np.exp(result.x[n_length_scales + 1])

                    self.length_scale = optimized_length_scales
                    self.signal_variance = optimized_signal_var
                    self.noise_variance = optimized_noise_var
                else:
                    self.length_scale = np.exp(result.x[0])
                    self.signal_variance = np.exp(result.x[1])
                    self.noise_variance = np.exp(result.x[2])

                self._log_hyperparameters(self.n_observations)
                self._invalidate_cache()

        except Exception as e:
            logger.error(f"GP Hyperparameter Optimization: ERROR - {e}")

    # ---------------------------------------------------------------------
    # SECTION: Data Export and Plotting
    # ---------------------------------------------------------------------

    def _get_plotting_data(self) -> Dict[str, Any]:
        """
        Gathers all data required for diagnostic plots into a dictionary.
        Updated to include Mean predictions for PCA grid.
        """
        data = {
            "principle_id": self.principle_id,
            "anisotropic": self.anisotropic,
            "feature_dim": self.feature_dim,
            "n_observations": self.n_observations,
            "max_inducing_points": self.max_inducing_points,
            "inducing_count": len(self.inducing_X)
            if self.inducing_X is not None
            else 0,
            "hyperparam_history": self.hyperparam_history,
            "X_data": [],
            "y_data": [],
            "inducing_X": [],
            "inducing_y": [],
            "y_pred": [],
            "y_std": [],
            "residuals": [],
            "rmse": None,
            "r2": None,
            "feature_relevance": [],
            "plot_1d_data": None,
            "plot_pca_data": None,
        }

        if self.n_observations < 3:
            return data

        # --- Basic Data ---
        X_data_np = np.array(self.X)
        y_data_np = np.array(self.y)
        data["X_data"] = X_data_np.tolist()
        data["y_data"] = y_data_np.tolist()
        if self.inducing_X is not None:
            data["inducing_X"] = self.inducing_X.tolist()
            data["inducing_y"] = self.inducing_y.tolist()

        # --- Predictions & Metrics ---
        try:
            preds = [self.predict(x) for x in X_data_np]
            y_pred = np.array([p[0] for p in preds])
            y_var = np.array([p[1] for p in preds])
            y_std = np.sqrt(y_var)
            y_std[y_std < 1e-9] = 1e-9

            residuals = (y_data_np - y_pred) / y_std

            mse = np.mean((y_data_np - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_data_np - y_pred) ** 2)
            ss_tot = np.sum((y_data_np - np.mean(y_data_np)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-9))

            data["y_pred"] = y_pred.tolist()
            data["y_std"] = y_std.tolist()
            data["residuals"] = residuals.tolist()
            data["rmse"] = float(rmse)
            data["r2"] = float(r2)

        except Exception as e:
            logger.error(f"Failed to generate predictions for plotting: {e}")
            return data

        # --- Feature Relevance ---
        if self.anisotropic:
            relevance = 1.0 / (self.length_scale + 1e-9)
            relevance = np.nan_to_num(relevance, nan=0.0)
            data["feature_relevance"] = relevance.tolist()

        # --- Feature Space Learning (1D) ---
        if self.feature_dim == 1:
            x_min, x_max = X_data_np[:, 0].min() - 1, X_data_np[:, 0].max() + 1
            xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            preds_1d = [self.predict(pt) for pt in xx]
            data["plot_1d_data"] = {
                "xx": xx.flatten().tolist(),
                "pred_mean": [p[0] for p in preds_1d],
                "pred_std": np.sqrt([p[1] for p in preds_1d]).tolist(),
            }

        # --- Feature Space Learning (PCA for >1D) ---
        elif self.feature_dim > 1 and PCA is not None:
            try:
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X_data_np)
                inducing_X_2d = pca.transform(self.inducing_X)
                explained_var = np.sum(pca.explained_variance_ratio_)

                # Create grid for surface plot
                x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
                y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25)
                )

                grid_2d = np.c_[xx.ravel(), yy.ravel()]
                grid_high_d = pca.inverse_transform(grid_2d)

                # Get both Mean and Variance for the grid
                preds_grid = [self.predict(pt) for pt in grid_high_d]
                pred_mean_grid = np.array([p[0] for p in preds_grid]).reshape(xx.shape)
                pred_std_grid = np.sqrt(np.array([p[1] for p in preds_grid])).reshape(
                    xx.shape
                )

                data["plot_pca_data"] = {
                    "X_2d": X_2d.tolist(),
                    "inducing_X_2d": inducing_X_2d.tolist(),
                    "explained_var": float(explained_var),
                    "xx": xx.tolist(),
                    "yy": yy.tolist(),
                    "pred_mean_grid": pred_mean_grid.tolist(),  # NEW: Added Mean
                    "pred_std_grid": pred_std_grid.tolist(),
                }

            except Exception as e:
                logger.error(f"PCA data generation failed: {e}")
                data["plot_pca_data"] = {"error": str(e)}

        return data

    def generate_full_diagnostic_report(
        self, save_path: str = "./gaussian_model_analysis.pdf"
    ):
        """
        Generates plotting data, saves it to JSON, and creates a
        single-page PDF report.
        """
        logger.info(f"Generating plotting data for {self.principle_id}...")
        plot_data = self._get_plotting_data()

        # Save the data to JSON
        json_path = os.path.join(
            self.log_dir, f"{self.principle_id}_plotting_data.json"
        )
        try:
            with open(json_path, "w") as f:
                json.dump(plot_data, f, indent=2)
            logger.info(f"Plotting data saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save plotting data to {json_path}: {e}")

        # Generate the report from the data
        logger.info(f"Generating diagnostic report PDF at {save_path}...")
        GaussianProcessModel._create_diagnostic_plots(plot_data, save_path)
        logger.info(f"GP Plotting: Compact diagnostic report saved to {save_path}")

    @staticmethod
    def plot_report_from_file(log_dir: str, principle_id: str, save_path: str):
        """
        Loads plotting data from a JSON file and generates the diagnostic PDF report.

        Args:
            log_dir: The base log directory (e.g., "./logs").
            principle_id: The ID of the principle (e.g., "P1").
            save_path: The full path to save the output PDF (e.g., "./report.pdf").
        """
        json_path = os.path.join(
            log_dir, "gaussian_process", f"{principle_id}_plotting_data.json"
        )

        if not os.path.exists(json_path):
            logger.error(f"Plotting data file not found: {json_path}")
            return

        logger.info(f"Loading plotting data from {json_path}...")
        try:
            with open(json_path, "r") as f:
                plot_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load plotting data from {json_path}: {e}")
            return

        # Generate the report from the loaded data
        logger.info(f"Generating diagnostic report PDF at {save_path}...")
        GaussianProcessModel._create_diagnostic_plots(plot_data, save_path)

    @staticmethod
    def _create_diagnostic_plots(data: Dict[str, Any], save_path: str):
        """
        Creates the diagnostic report.
        Uses a custom figure layout to allow a 3D subplot for the PCA view.
        """
        original_rc_params = plt.rcParams.copy()

        fig = plt.figure(figsize=(15, 12))

        # Title
        title = (
            f"Gaussian Process Model Report: {data['principle_id']}\n"
            f"N={data['n_observations']} | Inducing Pts: {data['inducing_count']}/{data['max_inducing_points']} | "
            f"Anisotropic: {data['anisotropic']} | Dim: {data['feature_dim']}"
        )
        fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")

        # Grid Spec for layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

        # 1. Model Fit (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        GaussianProcessModel._plot_model_diagnostics(ax1, data)

        # 2. Hyperparameter Evolution (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        GaussianProcessModel._plot_hyperparameter_evolution(ax2, data)

        # 3. Feature Space Learning (Bottom Left) - 3D PROJECTION if PCA data exists
        if (
            data["feature_dim"] > 1
            and data.get("plot_pca_data")
            and "error" not in data.get("plot_pca_data", {})
        ):
            ax3 = fig.add_subplot(gs[1, 0], projection="3d")
        else:
            ax3 = fig.add_subplot(gs[1, 0])
        GaussianProcessModel._plot_feature_space_learning(ax3, data)

        # 4. Relevance/Residuals (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        if data["anisotropic"]:
            GaussianProcessModel._plot_feature_relevance(ax4, data)
        else:
            GaussianProcessModel._plot_residuals(ax4, data)

        with PdfPages(save_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_model_diagnostics(ax: plt.Axes, data: Dict[str, Any]):
        """Plots Prediction vs. Actual with R^2 and RMSE annotations."""
        ax.set_title("Prediction vs. Actual Fit")
        if data["n_observations"] < 2 or not data["y_pred"]:
            ax.text(0.5, 0.5, s="No predictions to diagnose", ha="center", va="center")
            return

        y_data = np.array(data["y_data"])
        y_pred = np.array(data["y_pred"])
        r2 = data["r2"]
        rmse = data["rmse"]

        cmap = cm.get_cmap("magma")

        # Color scatter by actual value
        sc = ax.scatter(y_data, y_pred, alpha=0.7, s=25, c=y_data, cmap=cmap)

        lim_min = min(np.min(y_data), np.min(y_pred)) - 0.1
        lim_max = max(np.max(y_data), np.max(y_pred)) + 0.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="y=x")

        ax.set_xlabel("Actual Values (y)")
        ax.set_ylabel("Predicted Values ($\hat{y}$)")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.axis("equal")  # Ensure 1:1 aspect ratio

        # Add annotations
        stats_text = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$"
        ax.text(
            0.05,
            0.95,
            s=stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
        )

    @staticmethod
    def _plot_hyperparameter_evolution(ax: plt.Axes, data: Dict[str, Any]):
        """
        Plots hyperparameter evolution using Step plots to accurately show changes.
        Grouping: Signal Var & Noise Var (Left Axis), Length Scale (Right Axis).
        """
        ax.set_title("Hyperparameter Evolution", fontsize=12)
        history = data["hyperparam_history"]

        if len(history) < 2:
            ax.text(0.5, 0.5, s="Not enough history", ha="center", va="center")
            return

        n_obs = [h["n_observations"] for h in history]
        sig_var = [h["signal_variance"] for h in history]
        noise_var = [h["noise_variance"] for h in history]
        ls_mean = [h["length_scale_mean"] for h in history]

        # Colors
        color_sig = "#1f77b4"  # Blue
        color_noise = "#ff7f0e"  # Orange
        color_ls = "#2ca02c"  # Green

        # --- Left Axis: Variances (Signal and Noise) ---
        # Using steps-post because params stay constant until the next optimization
        ax.plot(
            n_obs,
            sig_var,
            label="Signal Var ($\sigma^2_f$)",
            color=color_sig,
            drawstyle="steps-post",
            linewidth=2,
        )
        ax.plot(
            n_obs,
            noise_var,
            label="Noise Var ($\sigma^2_n$)",
            color=color_noise,
            drawstyle="steps-post",
            linestyle="--",
            linewidth=2,
        )

        ax.set_xlabel("Number of Observations")
        ax.set_ylabel("Variance ($\sigma^2$)", color="black")
        ax.set_yscale("log")
        ax.tick_params(axis="y", which="both")

        # Grid for the main axis only
        ax.grid(True, which="major", linestyle=":", alpha=0.6)

        # --- Right Axis: Length Scale ---
        ax2 = ax.twinx()
        ax2.plot(
            n_obs,
            ls_mean,
            label="Mean Length Scale ($\ell$)",
            color=color_ls,
            drawstyle="steps-post",
            linewidth=2,
            alpha=0.8,
        )

        if data["anisotropic"]:
            ls_std = np.array([h["length_scale_std"] for h in history])
            ls_mean_np = np.array(ls_mean)
            # Use step='post' for fill_between to match the line
            ax2.fill_between(
                n_obs,
                ls_mean_np - ls_std,
                ls_mean_np + ls_std,
                step="post",
                color=color_ls,
                alpha=0.15,
            )

        ax2.set_ylabel("Length Scale ($\ell$)", color=color_ls)
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor=color_ls)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False,
        )

    @staticmethod
    def _plot_feature_relevance(ax: plt.Axes, data: Dict[str, Any]):
        """If anisotropic, plots the learned feature relevance (inverse length scale)."""

        # --- MODIFICATION: Update title ---
        ax.set_title("Relevance of First 20 Features")

        if not data["anisotropic"]:
            ax.text(0.5, 0.5, s="Only for Anisotropic models", ha="center", va="center")
            return
        if data["n_observations"] == 0:
            ax.text(0.5, 0.5, s="No data yet", ha="center", va="center")
            return

        relevance = np.array(data["feature_relevance"])
        max_relevance = np.max(relevance)
        if max_relevance < 1e-9:
            ax.text(0.5, 0.5, s="Relevance is zero", ha="center", va="center")
            return

        relevance_scaled = relevance / max_relevance

        num_features_to_show = min(data["feature_dim"], 20)

        indices_to_plot = np.arange(num_features_to_show)

        values_to_plot = relevance_scaled[indices_to_plot]
        labels_to_plot = [f"Feature {i}" for i in indices_to_plot]

        y_pos = np.arange(num_features_to_show)
        cmap = cm.get_cmap("magma")

        ax.barh(y_pos, values_to_plot, color=cmap(0.7), align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_to_plot)

        ax.set_xlabel("Relevance (Scaled to Max)")
        ax.set_ylabel("Feature Index")

        # --- MODIFICATION: Update title again (the original set_title was here) ---
        ax.set_title(f"Relevance of First {num_features_to_show} Features")

        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        # This will now plot Feature 0 at the top and Feature 19 at the bottom
        ax.invert_yaxis()

    @staticmethod
    def _plot_residuals(ax: plt.Axes, data: Dict[str, Any]):
        """Plots standardized residuals vs. predicted values."""
        ax.set_title("Residuals vs. Predicted")
        if data["n_observations"] < 2 or not data["residuals"]:
            ax.text(0.5, 0.5, s="No residuals to plot", ha="center", va="center")
            return

        y_pred = np.array(data["y_pred"])
        residuals = np.array(data["residuals"])
        cmap = cm.get_cmap("magma")

        # Color by absolute residual magnitude
        sc = ax.scatter(
            y_pred,
            residuals,
            alpha=0.7,
            s=25,
            c=np.abs(residuals),
            cmap=cmap,
            vmin=0,
            vmax=3,
        )

        ax.axhline(0, color="red", linestyle="--")
        ax.axhline(2, color="orange", linestyle=":", label="± 2 Std. Dev")
        ax.axhline(-2, color="orange", linestyle=":")
        ax.set_xlabel("Predicted Value ($\hat{y}$)")
        ax.set_ylabel("Standardized Residual")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.5)

    @staticmethod
    def _plot_feature_space_learning(ax: plt.Axes, data: Dict[str, Any]):
        """
        Visualizes the learned manifold.
        If Dim > 1: Uses 3D Surface plot (PCA1, PCA2, Predicted Value).
        """

        # Helper to safely render 2D text overlay on either 2D or 3D axes
        def safe_text(msg):
            if ax.name == "3d":
                # text2D places text relative to the axes (screen space), ignoring 3D rotation
                ax.text2D(
                    0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes
                )
            else:
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)

        if data["n_observations"] < 3:
            safe_text("Not enough data")
            return

        # --- 3D Plot Logic (PCA) ---
        if data["feature_dim"] > 1 and data.get("plot_pca_data"):
            pca_data = data["plot_pca_data"]

            # Check for error or missing grid data
            if "error" in pca_data or "pred_mean_grid" not in pca_data:
                safe_text(f"Data missing for 3D plot\n{pca_data.get('error', '')}")
                return

            # Unpack data
            X_2d = np.array(pca_data["X_2d"])
            ind_2d = np.array(pca_data["inducing_X_2d"])
            xx = np.array(pca_data["xx"])
            yy = np.array(pca_data["yy"])
            mu_grid = np.array(pca_data["pred_mean_grid"])

            # Actual Y values for scatter points
            y_data = np.array(data["y_data"])
            ind_y = np.array(data["inducing_y"])

            ax.set_title(
                f"Learned Manifold (PCA: {pca_data['explained_var'] * 100:.1f}%)", pad=0
            )

            # 1. Plot the Surface (The "GP Brain")
            # Color based on Z height (Mean)
            surf = ax.plot_surface(
                xx, yy, mu_grid, cmap="magma", alpha=0.4, linewidth=0, antialiased=False
            )

            # 2. Scatter Actual Observations
            ax.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                y_data,
                c="white",
                edgecolors="k",
                s=30,
                label="Observed",
                zorder=10,
            )

            # 3. Scatter Inducing Points (The "Anchors")
            ax.scatter(
                ind_2d[:, 0],
                ind_2d[:, 1],
                ind_y,
                c="red",
                marker="x",
                s=60,
                linewidth=2,
                label="Inducing Pts",
                zorder=11,
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("Outcome (y)")

            # Adjust view angle for better depth perception
            ax.view_init(elev=30, azim=-60)

            # Legend needs to be manually placed for 3D axes
            ax.legend(loc="upper right", fontsize=10)

        # --- 1D Plot Logic (Fallback) ---
        elif data["feature_dim"] == 1 and data.get("plot_1d_data"):
            ax.set_title("Learned Function (1D)")
            d1 = data["plot_1d_data"]
            xx = np.array(d1["xx"])
            mu = np.array(d1["pred_mean"])
            std = np.array(d1["pred_std"])

            ax.plot(xx, mu, "b-", label="Mean")
            ax.fill_between(
                xx, mu - 2 * std, mu + 2 * std, color="b", alpha=0.2, label="95% Conf"
            )
            ax.scatter(data["X_data"], data["y_data"], c="k", s=20, label="Obs")
            ax.scatter(
                data["inducing_X"],
                data["inducing_y"],
                c="r",
                marker="x",
                s=40,
                label="Inducing",
            )
            ax.legend()

        else:
            safe_text(f"No plot data available (Dim={data['feature_dim']})")


if __name__ == "__main__":
    PATH = "/Users/mellen/Desktop/PiEvo/reported_PiEvo_SPO_EXPORT_05"
    FILE_NAME = (
        "principle_topological_charge_density_wave_interference_plotting_data.json"
    )

    # ----- export ----
    principle_id = FILE_NAME.split("_plotting_data")[0]
    PDF_NAME = f"gp_report_{principle_id}.pdf"

    gp = GaussianProcessModel(
        principle_id=principle_id,
    )

    gp.plot_report_from_file(
        log_dir=os.path.join(PATH),
        principle_id=principle_id,
        save_path=os.path.join(PATH, PDF_NAME),
    )
