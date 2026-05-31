import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os
from cycler import cycler
from scipy.optimize import curve_fit


logger = logging.getLogger(__name__)


class TheoreticalAnalyzer:
    """
    Comprehensive analysis and plotting for PiEvo theoretical metrics.

    This class provides methods to visualize the dual uncertainty minimization
    process, regret decomposition, principle discovery dynamics, and convergence
    behavior as described in the theoretical framework.
    """

    def __init__(
        self,
        json_file_path: Optional[str] = None,
        flow_tracker_data: Optional[Dict] = None,
    ):
        """
        Initialize the analyzer with either a JSON file path or flow tracker data.

        Args:
            json_file_path: Path to saved JSON file with theoretical metrics
            flow_tracker_data: Dictionary with flow tracker data (alternative to file)
        """
        self.metrics_df = pd.DataFrame()
        self.quantities_df = pd.DataFrame()

        self.experiments_df = pd.DataFrame()
        self.principle_df = pd.DataFrame()
        self.hypothesis_df = pd.DataFrame()

        # Load data from JSON file or flow tracker data
        if json_file_path:
            self._load_from_json(json_file_path)
        else:
            logger.warning("No data provided for analysis")
            return

        # Set up plotting style
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """
        Set up a Matplotlib configuration that strictly follows the target academic chart style.
        """
        plt.style.use(
            "seaborn-v0_8-whitegrid"
        )  # Use a white background with a grid as a base

        # --- Color settings ---
        # Set colors in the order of GRPO, LUFFY, HPT for automatic use in plots
        color_palette = [
            "#5651a4",
            "#2F5597",
            "#C00000",
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
        ]

        plt.rcParams.update(
            {
                # --- Font settings ---
                "font.family": "serif",
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
                # --- Axes and tick settings ---
                "axes.edgecolor": "black",
                "axes.linewidth": 1.2,
                "xtick.color": "black",
                "ytick.color": "black",
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "xtick.major.width": 1,
                "ytick.major.width": 1,
                "xtick.top": True,
                "ytick.right": True,
                # --- Grid settings ---
                "grid.color": "#EAEAEA",
                "grid.linestyle": "-",
                "grid.linewidth": 0.8,
                "axes.axisbelow": True,
                # --- Global color cycle settings ---
                "axes.prop_cycle": cycler(
                    color=color_palette
                ),  # Global color configuration
                # --- Fix for minus sign display issue when saving figures ---
                "axes.unicode_minus": False,
            }
        )

    #
    def _load_from_json(self, json_file_path: str) -> None:
        """Load theoretical metrics from a saved JSON file."""
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            theoretical_metrics = data.get("logs", {}).get("theoretical_metrics", [])
            self._process_theoretical_logs(theoretical_metrics)

            theoretical_quantities = data.get("logs", {}).get(
                "theoretical_quantities", []
            )
            self._process_quantities_logs(theoretical_quantities)

        except Exception as e:
            logger.warning(
                f"Error loading JSON file {json_file_path}: {e} (Nothing will do for theoretical metrics plotting. Skipping)"
            )

    def _process_theoretical_logs(self, theoretical_logs: List[Dict]) -> None:
        """Process theoretical logs into a DataFrame."""
        if not theoretical_logs:
            logger.warning("No theoretical logs to process")
            return

        # Convert logs to DataFrame
        df = pd.DataFrame(theoretical_logs)

        # Convert timestamp to datetime if it exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Add round numbers if missing, before attempting to sort
        if "round_number" not in df.columns:
            df["round_number"] = range(1, len(df) + 1)

        # Handle potential None values and convert to appropriate types
        numeric_columns = [
            "round_number",
            "u_ep",
            "true_principle_posterior",
            "instantaneous_regret",
            "identification_regret",
            "ph_regret",
            "cumulative_regret",
            "average_regret",
            "principle_set_size",
            "anomaly_count",
            "anomaly_threshold",
            "information_gain",
            "cumulative_information",
            "candidate_pool_size",
            "information_ratio",
            "max_principle_belief",
            "principle_belief_entropy",
            "belief_concentration_ratio",
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Convert None to NaN and ensure numeric type
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where round_number could not be converted (is NaN)
        df.dropna(subset=["round_number"], inplace=True)
        # Ensure round_number is integer for clean axis labels
        df["round_number"] = df["round_number"].astype(int)

        # Sort by round number
        df = df.sort_values("round_number").reset_index(drop=True)

        self.metrics_df = df
        logger.info(f"Loaded {len(self.metrics_df)} theoretical metric records")

    def _process_quantities_logs(self, quantities_logs: List[Dict]) -> None:
        """Process theoretical logs into a DataFrame."""
        if not quantities_logs:
            logger.warning("No theoretical logs to process")
            return

        # Convert logs to DataFrame
        df = pd.DataFrame(quantities_logs)

        # Convert timestamp to datetime if it exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Add round numbers if missing, before attempting to sort
        if "round_number" not in df.columns:
            df["round_number"] = range(1, len(df) + 1)

        # Handle potential None values and convert to appropriate types
        numeric_columns = [
            "round_number",
            "information_budget_used",
            "information_budget_total",
            "num_principles",
            "candidate_pool_size",
            "max_belief_concentration",
            "principle_entropy",  # Add this column which is in the data structure
        ]

        # Only convert explicitly numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Convert None to NaN and ensure numeric type
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where round_number could not be converted (is NaN)
        df.dropna(subset=["round_number"], inplace=True)
        # Ensure round_number is integer for clean axis labels
        df["round_number"] = df["round_number"].astype(int)

        # Sort by round number
        df = df.sort_values("round_number").reset_index(drop=True)

        self.quantities_df = df
        logger.info(f"Loaded {len(self.quantities_df)} theoretical metric records")

        """
            Note: self.quantities_df = df:
        
            "theoretical_quantities": [
              {
                "timestamp": "2025-09-29T15:13:00.469642",
                "round_number": 2,
                "principle_to_hypothesis_uncertainty": null,
                "evidence_to_principle_uncertainty": 1.0986122886681096,
                "information_budget_used": 2.220446049250313e-16,
                "information_budget_total": 1.0986122886681098,
                "num_principles": 3,
                "principle_beliefs": {
                  "principle_hydrophobic_effect_driven_activity": 0.3333333333333333,
                  "principle_electrostatic_complementarity": 0.3333333333333333,
                  "principle_steric_complementarity": 0.3333333333333333
                },
                "principle_entropy": 1.0986122886681096,
                "generator_entropy": null,
                "candidate_pool_size": 2,
                "generator_satisfies_constraint": null,
                "max_belief_concentration": 1.0,
                "learning_phase": "prior_dominated"
              },
              {
                "timestamp": "2025-09-29T15:13:08.239806",
                "round_number": 2,
                "principle_to_hypothesis_uncertainty": null,
                "evidence_to_principle_uncertainty": 1.0986122886681096,
                "information_budget_used": 2.220446049250313e-16,
                "information_budget_total": 1.0986122886681098,
                "num_principles": 3,
                "principle_beliefs": {
                  "principle_hydrophobic_effect_driven_activity": 0.3333333333333333,
                  "principle_electrostatic_complementarity": 0.3333333333333333,
                  "principle_steric_complementarity": 0.3333333333333333
                },
                "principle_entropy": 1.0986122886681096,
                "generator_entropy": null,
                "candidate_pool_size": 2,
                "generator_satisfies_constraint": null,
                "max_belief_concentration": 1.0,
                "learning_phase": "prior_dominated"
              },
              {
                "timestamp": "2025-09-29T15:14:04.522310",
                "round_number": 4,
                "principle_to_hypothesis_uncertainty": null,
                "evidence_to_principle_uncertainty": 1.2333544568478765,
                "information_budget_used": -0.1347421681797667,
                "information_budget_total": 1.0986122886681098,
                "num_principles": 6,
                "principle_beliefs": {
                  "principle_hydrophobic_effect_driven_activity": 0.3233333333333334,
                  "principle_electrostatic_complementarity": 0.3233333333333334,
                  "principle_steric_complementarity": 0.32333333333333303,
                  "principle_hydrogen_bonding_network": 0.01,
                  "principle_molecular_polarity_balance": 0.01,
                  "principle_rotatable_bond_control": 0.01
                },
                "principle_entropy": 1.2333544568478765,
                "generator_entropy": null,
                "candidate_pool_size": 3,
                "generator_satisfies_constraint": null,
                "max_belief_concentration": 1.9400000000000006,
                "learning_phase": "transition"
              },
              {
                "timestamp": "2025-09-29T15:14:12.777955",
                "round_number": 4,
                "principle_to_hypothesis_uncertainty": null,
                "evidence_to_principle_uncertainty": 1.2333544568478765,
                "information_budget_used": -0.1347421681797667,
                "information_budget_total": 1.0986122886681098,
                "num_principles": 6,
                "principle_beliefs": {
                  "principle_hydrophobic_effect_driven_activity": 0.3233333333333334,
                  "principle_electrostatic_complementarity": 0.3233333333333334,
                  "principle_steric_complementarity": 0.32333333333333303,
                  "principle_hydrogen_bonding_network": 0.01,
                  "principle_molecular_polarity_balance": 0.01,
                  "principle_rotatable_bond_control": 0.01
                },
                "principle_entropy": 1.2333544568478765,
                "generator_entropy": null,
                "candidate_pool_size": 3,
                "generator_satisfies_constraint": null,
                "max_belief_concentration": 1.9400000000000006,
                "learning_phase": "transition"
              },
              {
                "timestamp": "2025-09-29T15:14:54.724196",
                "round_number": 6,
                "principle_to_hypothesis_uncertainty": null,
                "evidence_to_principle_uncertainty": 1.385514172632836,
                "information_budget_used": -0.28690188396472616,
                "information_budget_total": 1.0986122886681098,
                "num_principles": 9,
                "principle_beliefs": {
                  "principle_hydrophobic_effect_driven_activity": 0.31151102606170605,
                  "principle_electrostatic_complementarity": 0.3115110260617081,
                  "principle_steric_complementarity": 0.3115110260617081,
                  "principle_hydrogen_bonding_network": 0.011822307271625861,
                  "principle_molecular_polarity_balance": 0.011822307271625851,
                  "principle_rotatable_bond_control": 0.011822307271625851,
                  "principle_metal_coordination": 0.01,
                  "principle_pi_stacking_interactions": 0.01,
                  "principle_solvent_exposure_minimization": 0.01
                },
                "principle_entropy": 1.385514172632836,
                "generator_entropy": null,
                "candidate_pool_size": 3,
                "generator_satisfies_constraint": null,
                "max_belief_concentration": 2.803599234555373,
                "learning_phase": "transition"
              },
              ...
            ]
        
        """

    def _safe_plot(self, ax, x, y, is_scatter=False, *args, **kwargs) -> bool:
        """Safely plot data, handling NaN and None values."""
        try:
            if y is None or len(y) == 0:
                return False

            # Convert to numpy arrays and handle NaN values
            y_array = np.array(y, dtype=float)
            x_array = np.array(x) if x is not None else np.arange(len(y))

            # Remove NaN values from y and corresponding x
            valid_mask = ~np.isnan(y_array)
            if not np.any(valid_mask):
                return False

            x_valid = x_array[valid_mask]
            y_valid = y_array[valid_mask]

            if len(x_valid) == 0:
                return False
            if is_scatter:
                ax.scatter(x_valid, y_valid, *args, **kwargs)
            else:
                ax.plot(x_valid, y_valid, *args, **kwargs)
            return True
        except Exception as e:
            logger.warning(f"Error plotting data: {e}")
            return False

    # ==================================== FITTING FUNCTIONS ====================================

    @staticmethod
    def _sqrt_func(t, a, b):
        """Model for O(sqrt(t)) growth. Adding an intercept 'b' for a better fit."""
        return a * np.sqrt(t) + b

    @staticmethod
    def _log_func(t, a, b):
        """Model for O(log(t)) growth."""
        # Add a small epsilon to avoid log(0) for round 0 if it ever occurs
        return a * np.log(t + 1e-9) + b

    @staticmethod
    def _inv_sqrt_func(t, a, b):
        """Model for O(1/sqrt(t)) decay. Adding an intercept 'b' for a better fit."""
        # Add a small epsilon to avoid division by zero
        return a / np.sqrt(t + 1e-9) + b

    @staticmethod
    def _log_div_t_func(t, a, b):
        """Model for O(log(t)/t) decay."""
        return (a * np.log(t + 1e-9) + b) / (t + 1e-9)

    def _fit_and_select_best_model(
        self, x_data, y_data, models: Dict
    ) -> Optional[Tuple[str, np.ndarray, float, callable]]:
        """
        Fit multiple models to the data and select the best one based on R-squared.

        Args:
            x_data: The independent variable data (rounds).
            y_data: The dependent variable data (regret).
            models: A dictionary mapping model names to fitting functions.

        Returns:
            A tuple containing:
            - best_model_name (str)
            - best_params (np.ndarray)
            - best_r2 (float)
            - best_func (callable)
            Or None if no model could be fitted.
        """
        best_fit = None
        best_r2 = -np.inf

        for name, func in models.items():
            try:
                # Perform the curve fitting
                params, _ = curve_fit(func, x_data, y_data, maxfev=5000)

                # Calculate R-squared value for the fit
                y_pred = func(x_data, *params)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

                # Avoid division by zero if all y values are the same
                if ss_tot == 0:
                    r2 = 1.0 if ss_res == 0 else 0.0
                else:
                    r2 = 1 - (ss_res / ss_tot)

                # If this model is better, store it
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = (name, params, r2, func)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Could not fit model '{name}': {e}")
                continue

        return best_fit

    # ==================================== PLOTTING FUNCTIONS ====================================

    def plot_dual_uncertainty_minimization(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the core dual uncertainty minimization process.
        """
        if self.metrics_df.empty:
            logger.warning("No data available for dual uncertainty minimization plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Dual Uncertainty Minimization Process")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        rounds = self.metrics_df["round_number"]

        # Plot U^EP_t (Evidence-to-Principle uncertainty)
        self._safe_plot(
            ax1,
            rounds,
            self.metrics_df.get("u_ep"),
            linewidth=2,
            label="$U^{EP}_t$",
            marker="o",
            markersize=4,
        )

        ax1.set_ylabel("$U^{EP}_t$", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Dual Uncertainty Minimization Process", fontsize=14, fontweight="bold"
        )
        ax1.grid(True)
        ax1.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved dual uncertainty plot to {save_path}")

        return fig

    def plot_regret_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cumulative and average regret, fitting them to theoretical sub-linear models.
        """
        if self.metrics_df.empty:
            logger.warning("No data available for regret analysis plot")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            axes[0].set_title("Cumulative Regret")
            axes[1].set_title("Average Regret")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        rounds = self.metrics_df["round_number"]

        # --- Plot Cumulative Regret with Scientific Fitting ---
        if "cumulative_regret" in self.metrics_df.columns:
            valid_mask = self.metrics_df["cumulative_regret"].notna()
            if valid_mask.any():
                valid_rounds = rounds[valid_mask].to_numpy()
                valid_cumulative = self.metrics_df["cumulative_regret"][
                    valid_mask
                ].to_numpy()

                ax1.plot(
                    valid_rounds,
                    valid_cumulative,
                    linewidth=2,
                    label="Cumulative Regret",
                    marker="o",
                    markersize=4,
                )

                cumulative_models = {
                    "$O(\\sqrt{t})$": self._sqrt_func,
                    "$O(\\log t)$": self._log_func,
                }

                if len(valid_rounds) > 2:
                    best_fit = self._fit_and_select_best_model(
                        valid_rounds, valid_cumulative, cumulative_models
                    )
                    if best_fit:
                        name, params, r2, func = best_fit
                        fit_label = f"Best Fit: {name} ($R^2={r2:.2f}$)"
                        ax1.plot(
                            valid_rounds,
                            func(valid_rounds, *params),
                            linestyle="--",
                            linewidth=2,
                            label=fit_label,
                        )
        else:
            ax1.text(
                0.5,
                0.5,
                "No cumulative regret data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        ax1.set_ylabel("Cumulative Regret", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title("Cumulative Regret Analysis", fontsize=14, fontweight="bold")

        # --- Plot Average Regret with Scientific Fitting ---
        if "average_regret" in self.metrics_df.columns:
            valid_mask = self.metrics_df["average_regret"].notna()
            if valid_mask.any():
                valid_rounds = rounds[valid_mask].to_numpy()
                valid_average = self.metrics_df["average_regret"][valid_mask].to_numpy()

                ax2.plot(
                    valid_rounds,
                    valid_average,
                    linewidth=2,
                    label="Average Regret",
                    marker="s",
                    markersize=4,
                )

                average_models = {
                    "$O(1/\\sqrt{t})$": self._inv_sqrt_func,
                    "$O(\\log t / t)$": self._log_div_t_func,
                }

                if len(valid_rounds) > 2:
                    best_fit = self._fit_and_select_best_model(
                        valid_rounds, valid_average, average_models
                    )
                    if best_fit:
                        name, params, r2, func = best_fit
                        fit_label = f"Best Fit: {name} ($R^2={r2:.2f}$)"
                        ax2.plot(
                            valid_rounds,
                            func(valid_rounds, *params),
                            linestyle="--",
                            linewidth=2,
                            label=fit_label,
                        )
        else:
            ax2.text(
                0.5,
                0.5,
                "No average regret data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        ax2.set_ylabel("Average Regret", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title("Average Regret Analysis", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved regret analysis plot to {save_path}")

        return fig

    def plot_principle_discovery_dynamics(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot principle discovery dynamics and belief evolution.
        """
        if self.metrics_df.empty:
            logger.warning("No data available for principle discovery dynamics plot")
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            for ax_sub in ax.flatten():
                ax_sub.text(0.5, 0.5, "No data available", ha="center", va="center")
            fig.suptitle("Principle Discovery and Belief Dynamics")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        rounds = self.metrics_df["round_number"]

        # Plot 1: Size of working principle set |P_t|
        if "principle_set_size" in self.metrics_df.columns:
            plotted = self._safe_plot(
                ax1,
                rounds,
                self.metrics_df["principle_set_size"],
                linewidth=2,
                marker="o",
            )
            if plotted:
                ax1.step(
                    rounds,
                    self.metrics_df["principle_set_size"],
                    linewidth=2,
                    where="post",
                    alpha=0.7,
                )
                ax1.fill_between(
                    rounds,
                    0,
                    self.metrics_df["principle_set_size"],
                    step="post",
                    alpha=0.3,
                )
                ax1.set_ylabel("Number of Principles", fontsize=12)
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Principle set size data\\nnot available",
                    ha="center",
                    va="center",
                )
        else:
            ax1.text(
                0.5,
                0.5,
                "Principle set size column\\nnot found",
                ha="center",
                va="center",
            )

        ax1.set_title("Principle Set Size Evolution", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot 2: True principle posterior (if available)
        if "true_principle_posterior" in self.metrics_df.columns:
            plotted = self._safe_plot(
                ax2,
                rounds,
                self.metrics_df["true_principle_posterior"],
                linewidth=2,
                marker="o",
            )
            if plotted:
                ax2.axhline(
                    y=1.0, linestyle="--", alpha=0.5, label="Perfect identification"
                )
                ax2.set_ylabel("Posterior Probability", fontsize=12)
                ax2.legend()
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "True principle data\\nnot available",
                    ha="center",
                    va="center",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "True principle posterior\\ncolumn not found",
                ha="center",
                va="center",
            )

        ax2.set_title("True Principle Identification", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Belief concentration
        if self._safe_plot(
            ax3,
            rounds,
            self.metrics_df.get("max_principle_belief"),
            linewidth=2,
            label="Max Belief",
            marker="s",
        ) or self._safe_plot(
            ax3,
            rounds,
            self.metrics_df.get("belief_concentration_ratio"),
            linewidth=2,
            label="Concentration Ratio",
            marker="^",
        ):
            ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Uniform")
            ax3.legend()
        else:
            ax3.text(
                0.5,
                0.5,
                "Belief concentration data\\nnot available",
                ha="center",
                va="center",
            )

        ax3.set_ylabel("Belief Measures", fontsize=12)
        ax3.set_title("Belief Concentration", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Learning phase evolution
        if "learning_phase" in self.metrics_df.columns:
            phase_mapping = {
                "exploration": 0,
                "transition": 1,
                "exploitation": 2,
                "unknown": -1,
            }
            phase_numeric = self.metrics_df["learning_phase"].map(phase_mapping)

            valid_mask = phase_numeric >= 0
            if valid_mask.any():
                valid_rounds = rounds[valid_mask]
                valid_phases = phase_numeric[valid_mask]
                ax4.step(valid_rounds, valid_phases, linewidth=2, where="post")
                ax4.fill_between(valid_rounds, 0, valid_phases, step="post", alpha=0.3)
                ax4.set_yticks([0, 1, 2])
                ax4.set_yticklabels(["Exploration", "Transition", "Exploitation"])
            else:
                ax4.text(0.5, 0.5, "No valid learning phases", ha="center", va="center")
        else:
            ax4.text(
                0.5, 0.5, "Learning phase column\\nnot found", ha="center", va="center"
            )

        ax4.set_ylabel("Learning Phase", fontsize=12)
        ax4.set_title("Learning Phase Evolution", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        for ax in axes.flatten():
            ax.set_xlabel("Round t", fontsize=12)

        plt.suptitle(
            "Principle Discovery and Belief Dynamics", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved principle discovery plot to {save_path}")

        return fig

    def plot_information_gain(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot information gain in a single figure using line plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Information Gain Analysis", fontsize=14, fontweight="bold")

        if self._safe_plot(
            ax,
            self.metrics_df.get("round_number"),
            self.metrics_df.get("information_gain"),
            linewidth=2,
            label="Information Gain $I_t$",
            marker="o",
            markersize=4,
        ):
            ax.set_ylabel("Information Gain $I_t$", fontsize=12, fontweight="bold")
            ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No information gain data available (will start later)",
                ha="center",
                va="center",
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved information gain plot to {save_path}")
        return fig

    def plot_information_ratio(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot information ratio in a single figure using line plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Information Ratio Analysis", fontsize=14, fontweight="bold")

        if self._safe_plot(
            ax,
            self.metrics_df.get("round_number"),
            self.metrics_df.get("information_ratio"),
            linewidth=2,
            label="Information Ratio $\\Gamma(h_t)$",
            marker="d",
            markersize=4,
        ):
            ax.set_ylabel(
                "Information Ratio $\\Gamma(h_t)$", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No information ratio data available",
                ha="center",
                va="center",
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved information ratio plot to {save_path}")
        return fig

    def plot_learning_phase_distribution(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning phase distribution using bar plot.
        """
        # TODO: Change this entire function into the line plot similar to others, by using `round_number` and `belief_concentration_ratio`.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Learning Phase Distribution", fontsize=14, fontweight="bold")

        if (
            "learning_phase" in self.metrics_df.columns
            and not self.metrics_df["learning_phase"].empty
        ):
            phase_counts = self.metrics_df["learning_phase"].value_counts()
            phase_counts = phase_counts[phase_counts.index != "unknown"]
            if not phase_counts.empty:
                bars = ax.bar(phase_counts.index, phase_counts.values)
                ax.set_ylabel("Count", fontsize=12, fontweight="bold")
                ax.set_xlabel("Learning Phase", fontsize=12, fontweight="bold")

                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
                ax.grid(True, alpha=0.3, axis="y")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No learning phase data available",
                    ha="center",
                    va="center",
                )
        else:
            ax.text(
                0.5, 0.5, "No learning phase data available", ha="center", va="center"
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved learning phase distribution plot to {save_path}")
        return fig

    def plot_information_budget_utilization(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot information budget utilization over time using seaborn.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for information budget utilization plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Information Budget Utilization")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if required columns exist
        if (
            "round_number" not in self.quantities_df.columns
            or "information_budget_used" not in self.quantities_df.columns
            or "information_budget_total" not in self.quantities_df.columns
        ):
            ax.text(
                0.5,
                0.5,
                "Required columns not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Information Budget Utilization")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create a seaborn line plot for information budget used
        sns.lineplot(
            data=self.quantities_df,
            x="round_number",
            y="information_budget_used",
            ax=ax,
            marker="o",
            label="Budget Used",
            linewidth=2,
        )

        # Also plot the total information budget if it's variable
        if self.quantities_df["information_budget_total"].nunique() > 1:
            sns.lineplot(
                data=self.quantities_df,
                x="round_number",
                y="information_budget_total",
                ax=ax,
                marker="s",
                label="Budget Total",
                linewidth=2,
            )
        else:
            # If total budget is constant, show as a horizontal line
            total_budget = self.quantities_df["information_budget_total"].iloc[0]
            ax.axhline(
                y=total_budget,
                linestyle="--",
                alpha=0.7,
                label="Budget Total",
                color="red",
            )

        ax.set_title(
            "Information Budget Utilization Over Time", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax.set_ylabel("Information Budget", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved information budget utilization plot to {save_path}")

        return fig

    def plot_num_principles_evolution(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot number of principles discovered over time using seaborn.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for number of principles evolution plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Number of Principles Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if required columns exist
        if (
            "round_number" not in self.quantities_df.columns
            or "num_principles" not in self.quantities_df.columns
        ):
            ax.text(
                0.5,
                0.5,
                "Required columns not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Number of Principles Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create a seaborn line plot for number of principles
        sns.lineplot(
            data=self.quantities_df,
            x="round_number",
            y="num_principles",
            ax=ax,
            marker="o",
            linewidth=2,
        )

        ax.set_title(
            "Number of Principles Evolution Over Time", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Principles", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add value annotations for each point
        for i, row in self.quantities_df.iterrows():
            ax.annotate(
                f"{int(row['num_principles'])}",
                (row["round_number"], row["num_principles"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved number of principles evolution plot to {save_path}")

        return fig

    def plot_candidate_pool_size_evolution(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot candidate pool size evolution over time using seaborn.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for candidate pool size evolution plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Candidate Pool Size Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if required columns exist
        if (
            "round_number" not in self.quantities_df.columns
            or "candidate_pool_size" not in self.quantities_df.columns
        ):
            ax.text(
                0.5,
                0.5,
                "Required columns not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Candidate Pool Size Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create a seaborn line plot for candidate pool size
        sns.lineplot(
            data=self.quantities_df,
            x="round_number",
            y="candidate_pool_size",
            ax=ax,
            marker="o",
            linewidth=2,
        )

        ax.set_title(
            "Candidate Pool Size Evolution Over Time", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax.set_ylabel("Candidate Pool Size", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved candidate pool size evolution plot to {save_path}")

        return fig

    def plot_principle_entropy_evolution(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot principle entropy evolution over time using seaborn.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for principle entropy evolution plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Entropy Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if required columns exist
        if "round_number" not in self.quantities_df.columns:
            ax.text(
                0.5,
                0.5,
                "Round number column not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Entropy Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Check if principle_entropy column exists in the dataframe
        if "principle_entropy" not in self.quantities_df.columns:
            ax.text(
                0.5,
                0.5,
                "No principle_entropy column available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Entropy Evolution")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create a seaborn line plot for principle entropy
        sns.lineplot(
            data=self.quantities_df,
            x="round_number",
            y="principle_entropy",
            ax=ax,
            marker="o",
            linewidth=2,
        )

        ax.set_title(
            "Principle Entropy Evolution Over Time", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax.set_ylabel("Principle Entropy", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved principle entropy evolution plot to {save_path}")

        return fig

    def plot_principle_belief_heatmap(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap for principle belief distributions using seaborn.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for principle belief heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Belief Heatmap")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Check if principle_beliefs column exists in the dataframe
        if "principle_beliefs" not in self.quantities_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No principle beliefs column available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Belief Heatmap")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Extract principle names from the first row, assuming they're consistent
        first_beliefs = (
            self.quantities_df["principle_beliefs"].iloc[0]
            if not self.quantities_df.empty
            else {}
        )

        if not isinstance(first_beliefs, dict):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Principle beliefs are not in dictionary format",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Belief Heatmap")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Get all unique principle names
        all_principles = set()
        for beliefs in self.quantities_df["principle_beliefs"]:
            if isinstance(beliefs, dict):
                all_principles.update(beliefs.keys())
        all_principles = sorted(list(all_principles))

        # Create a matrix of principle beliefs over time
        belief_matrix = []
        valid_rounds = []

        for idx, row in self.quantities_df.iterrows():
            if isinstance(row["principle_beliefs"], dict):
                beliefs_row = []
                for principle in all_principles:
                    beliefs_row.append(row["principle_beliefs"].get(principle, 0.0))
                belief_matrix.append(beliefs_row)
                valid_rounds.append(row["round_number"])

        if not belief_matrix:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No valid principle belief data to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Principle Belief Heatmap")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create a DataFrame suitable for seaborn heatmap
        belief_df = pd.DataFrame(
            belief_matrix, columns=all_principles, index=valid_rounds
        )
        belief_df.index.name = "Round Number"

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            belief_df.T,
            xticklabels=True,
            yticklabels=True,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Belief Probability"},
        )

        ax.set_title(
            "Principle Belief Distribution Heatmap", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Round t", fontsize=12, fontweight="bold")
        ax.set_ylabel("Principles", fontsize=12, fontweight="bold")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved principle belief heatmap to {save_path}")

        return fig

    def plot_dynamic_principle_bubbles(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a dynamic bubble plot showing principle beliefs as bubbles with size proportional to belief values.
        """
        if self.quantities_df.empty:
            logger.warning("No data available for principle bubble plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Dynamic Principle Belief Bubbles")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Check if principle_beliefs column exists
        if "principle_beliefs" not in self.quantities_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No principle beliefs column available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Dynamic Principle Belief Bubbles")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Get all unique principle names
        all_principles = set()
        for beliefs in self.quantities_df["principle_beliefs"]:
            if isinstance(beliefs, dict):
                all_principles.update(beliefs.keys())
        all_principles = sorted(list(all_principles))

        if not all_principles:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No principle data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Dynamic Principle Belief Bubbles")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Create the bubble plot for the last round (or a specific round)
        # For a static version of the dynamic bubble plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use the last round of data to create the bubble plot
        last_round_data = self.quantities_df.iloc[-1]
        principle_beliefs = last_round_data["principle_beliefs"]

        if not isinstance(principle_beliefs, dict):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Principle beliefs are not in dictionary format",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Dynamic Principle Belief Bubbles")
            if save_path:
                plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            return fig

        # Prepare data for the bubble plot
        principles = list(principle_beliefs.keys())
        values = list(principle_beliefs.values())

        # To position bubbles in a visually pleasing way, we'll distribute them in a circle
        n_principles = len(principles)
        angles = np.linspace(0, 2 * np.pi, n_principles, endpoint=False)

        # Normalize values to a reasonable bubble size range
        max_size = 1000
        min_size = 50
        normalized_values = (np.array(values) - min(values)) / (
            max(values) - min(values) if max(values) != min(values) else 1
        )
        bubble_sizes = min_size + normalized_values * (max_size - min_size)

        # Calculate positions in a circle
        x_pos = np.cos(angles) * 3
        y_pos = np.sin(angles) * 3

        # Create scatter plot with bubbles
        scatter = ax.scatter(
            x_pos,
            y_pos,
            s=bubble_sizes,
            c=range(n_principles),
            cmap="tab20",
            alpha=0.6,
            edgecolors="black",
        )

        # Add labels to bubbles
        for i, (x, y, principle, value) in enumerate(
            zip(x_pos, y_pos, principles, values)
        ):
            ax.text(
                x,
                y,
                f"{principle.split('_')[1:][0] if '_' in principle else principle[:10]}\n{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.set_xlim(min(x_pos) - 1, max(x_pos) + 1)
        ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
        ax.set_title(
            f"Principle Beliefs Bubble Plot (Round {last_round_data['round_number']})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add a colorbar legend
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Principle Index", fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format="pdf", facecolor="white")
            logger.info(f"Saved principle bubble plot to {save_path}")

        return fig

    def create_animated_principle_bubbles(
        self, save_path: Optional[str] = None
    ) -> None:
        """
        Create an animated MP4 showing the dynamic evolution of principle beliefs ranked as a bar chart leaderboard.
        Principles are ordered by belief value from highest to lowest, with smooth transitions between states.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            logger.warning("matplotlib.animation not available")
            return

        if self.quantities_df.empty:
            logger.warning("No data available for animated principle bar plot")
            return

        if "principle_beliefs" not in self.quantities_df.columns:
            logger.warning("No principle beliefs column available")
            return

        # Get all unique principle names across all rounds
        all_principles = set()
        for beliefs in self.quantities_df["principle_beliefs"]:
            if isinstance(beliefs, dict):
                all_principles.update(beliefs.keys())
        all_principles = sorted(list(all_principles))

        if not all_principles:
            logger.warning("No principle data available")
            return

        # Prepare data for animation - group by round number to get single representative per round
        round_groups = self.quantities_df.groupby("round_number")
        unique_rounds = sorted(round_groups.groups.keys())

        principle_data_by_round = {}
        for round_num in unique_rounds:
            round_data = round_groups.get_group(round_num).iloc[
                0
            ]  # Take the first instance of each round
            if isinstance(round_data["principle_beliefs"], dict):
                principle_data_by_round[round_num] = round_data["principle_beliefs"]

        if not principle_data_by_round:
            logger.warning("No principle belief data available")
            return

        # Set up the figure and axis for bar chart
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create a colormap for consistent coloring
        colormap = plt.cm.viridis
        principle_colors = {
            p: colormap(i / len(all_principles)) for i, p in enumerate(all_principles)
        }

        def animate(frame_idx):
            ax.clear()

            # Cycle through actual rounds
            round_num = (
                unique_rounds[frame_idx % len(unique_rounds)] if unique_rounds else 0
            )
            if round_num in principle_data_by_round:
                current_data = principle_data_by_round[round_num]

                # Extract and sort principles by belief value (highest to lowest)
                principle_values = [
                    (p, current_data.get(p, 0.0))
                    for p in all_principles
                    if p in current_data
                ]
                principle_values = sorted(
                    principle_values, key=lambda x: x[1], reverse=True
                )  # Sort by value (descending)

                if principle_values:
                    principles, values = zip(*principle_values)
                    n_principles = len(principles)

                    # Create ordered bar chart
                    y_pos = range(
                        n_principles
                    )  # Y positions from top to bottom (highest to lowest)

                    # Get colors based on the ordered principles
                    colors = [principle_colors[p] for p in principles]

                    # Create horizontal bars
                    bars = ax.barh(
                        y_pos,
                        values,
                        color=colors,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                    )

                    # Add value labels on bars
                    for i, (principle, value) in enumerate(principle_values):
                        # Shorten principle name for display
                        short_name = (
                            principle.replace("principle_", "")
                            .replace("_", " ")
                            .title()
                        )
                        if len(short_name) > 20:  # Truncate long names
                            short_name = short_name[:20] + "..."

                        # Position text inside the bar if it's long enough, otherwise outside
                        if (
                            value > 0.05
                        ):  # If the bar is long enough to show text inside
                            ax.text(
                                value / 2,
                                i,
                                f"{short_name}\n{value:.3f}",
                                ha="center",
                                va="center",
                                fontsize=10,
                                color="white",
                                weight="bold",
                            )
                        else:  # Show text outside the bar
                            ax.text(
                                value + 0.01,
                                i,
                                f"{short_name}\n{value:.3f}",
                                ha="left",
                                va="center",
                                fontsize=10,
                                color="black",
                                weight="bold",
                            )

                    ax.set_xlabel("Belief Value", fontsize=14, fontweight="bold")
                    ax.set_ylabel("Ranked Principles", fontsize=14, fontweight="bold")
                    ax.set_title(
                        f"Principle Belief Leaderboard (Round {round_num})",
                        fontsize=16,
                        fontweight="bold",
                        pad=20,
                    )

                    # Set y-axis labels to be the short names of principles
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(
                        [
                            p.replace("principle_", "").replace("_", " ").title()[:20]
                            for p in principles
                        ]
                    )

                    # Invert y-axis so highest values are at the top
                    ax.invert_yaxis()

                    # Set x-axis limits to accommodate all bars and text
                    ax.set_xlim(0, max(values) * 1.2 if values else 1)

                    ax.grid(axis="x", alpha=0.3)
                    ax.set_axisbelow(True)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No active principles",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16,
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data for this round",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            return ax.patches  # Return the bars for animation

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(unique_rounds),
            interval=500,
            blit=False,
            repeat=True,
        )

        # Save as MP4 for better quality and smoother animation
        if save_path:
            video_path = (
                save_path.replace(".pdf", ".mp4")
                if save_path.endswith(".pdf")
                else save_path
            )
            if not video_path.endswith(".mp4"):
                video_path += ".mp4"

            try:
                # Try to save as MP4 using ffmpeg
                anim.save(video_path, writer="ffmpeg", fps=2, dpi=300)
                logger.info(f"Saved animated principle bar chart video to {video_path}")
            except Exception as e:
                logger.warning(
                    f"Could not save as MP4: {e}. Trying alternative format..."
                )
                try:
                    # Fallback to GIF - but make sure we don't save each frame as an image and then combine
                    gif_path = video_path.replace(".mp4", ".gif")
                    anim.save(gif_path, writer="pillow", fps=1)
                    logger.info(f"Saved animated principle bar chart GIF to {gif_path}")
                except Exception as e2:
                    logger.error(f"Could not save animation: {e2}")
        else:
            # If no save path provided, save as default file
            try:
                anim.save(
                    "animated_principle_leaderboard.mp4",
                    writer="ffmpeg",
                    fps=1,
                    dpi=100,
                )
                logger.info(
                    "Saved animated principle bar chart video to animated_principle_leaderboard.mp4"
                )
            except Exception:
                try:
                    anim.save(
                        "animated_principle_leaderboard.gif", writer="pillow", fps=1
                    )
                    logger.info(
                        "Saved animated principle bar chart GIF to animated_principle_leaderboard.gif"
                    )
                except Exception as e:
                    logger.error(f"Could not save animation: {e}")

        plt.close(fig)

    def generate_comprehensive_report(
        self, output_dir: str = "theoretical_analysis"
    ) -> Dict[str, str]:
        """
        Generate a comprehensive analysis report with all theoretical plots.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_files = {}
        plot_functions = {
            "01_dual_uncertainty_minimization": self.plot_dual_uncertainty_minimization,
            "02_regret_analysis": self.plot_regret_analysis,
            "03_principle_discovery_dynamics": self.plot_principle_discovery_dynamics,
            "04_information_gain": self.plot_information_gain,
            "05_information_ratio": self.plot_information_ratio,
            "06_learning_phase_distribution": self.plot_learning_phase_distribution,
            "07_information_budget_utilization": self.plot_information_budget_utilization,
            "08_num_principles_evolution": self.plot_num_principles_evolution,
            "09_candidate_pool_size_evolution": self.plot_candidate_pool_size_evolution,
            "10_principle_entropy_evolution": self.plot_principle_entropy_evolution,
            "11_principle_belief_heatmap": self.plot_principle_belief_heatmap,
            "12_dynamic_principle_bubbles": self.plot_dynamic_principle_bubbles,
        }

        for name, func in plot_functions.items():
            logger.info(f"Generating {name} plot...")
            try:
                fig = func()
                path = os.path.join(output_dir, f"{name}.pdf")
                fig.savefig(path, dpi=300, format="pdf", facecolor="white")
                plt.close(fig)
                plot_files[name] = path
            except Exception as e:
                logger.error(f"Failed to generate plot {name}: {e}")

        # Handle the animated GIF separately as it doesn't return a figure
        logger.info("Generating animated_principle_bubbles GIF...")
        try:
            gif_path = os.path.join(output_dir, "13_animated_principle_bubbles.gif")
            self.create_animated_principle_bubbles(save_path=gif_path)
            plot_files["13_animated_principle_bubbles"] = gif_path
        except Exception as e:
            logger.error(f"Failed to generate animated principle bubbles: {e}")

        logger.info(
            f"Comprehensive theoretical analysis report generated in: {output_dir}"
        )
        return plot_files


if __name__ == "__main__":
    track = TheoreticalAnalyzer(
        json_file_path="../../03_sigma=0d001/pievo_metric_summary.json"
    )
    track.generate_comprehensive_report(
        output_dir="../../03_sigma=0d001/theoretical_analysis_output"
    )
