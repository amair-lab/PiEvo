import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import math
import matplotlib.collections as mcoll


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PrincipleLog:
    round_number: int
    num_principles: int
    num_anomalies: int
    anomaly_threshold: float
    anomalies: List[
        Dict[str, Any]
    ]  # [{"hypothesis": str, "expected": float, "actual": float}]
    principle_beliefs: Dict[str, float]  # mapping principle_id -> posterior prob p_t(P)
    principle_uncertainty_uep: float  # posterior entropy H(P | H_{t-1})
    guidance_message: str
    guidance_type: str


@dataclass
class HypothesisLog:
    round_number: int
    selected_principle_id: Optional[str]
    selected_principle_text: str
    principle_beliefs: Dict[str, float]
    principle_uncertainty_uep: float
    selection_method: str  # "thompson_sampling", "map", "fallback", etc.
    guidance_message: str
    guidance_type: str


@dataclass
class ExperimentLog:
    """
    Experiment-level logging.

    NOTE: a few additional fields are added compared to your original snippet.
    These are minimal, natural pieces of information the algorithm can compute at
    experiment time and are required to compute regret decomposition, information
    ratios, and other theoretical metrics precisely, while keeping the PiEvo
    runtime decoupled from the FlowTracker.

    Required / optional fields that PiEvo should populate when recording an experiment:
      - round_number, phase, selected_principle_id, selected_hypothesis, selection_method
      - observed_reward: actual scalar reward observed (r(h_t, P^*) if P^* known, or just observed value)
      - selected_hypothesis_expected_rewards: mapping principle_id -> E[r(h_t, P)] (expected reward of selected hypothesis under each principle)
      - principle_optimal_values: mapping principle_id -> v(P) = max_h r(h, P)
      - information_ratios: mapping hypothesis -> {"regret": float, "info_gain": float, "ratio": float}
    If some of these fields are not available for a round, FlowTracker will compute a partial set of metrics.
    """

    round_number: int
    phase: str  # "warm_up", "ids", etc.
    # observed scalar reward (the actual outcome's reward according to the environment)
    observed_reward: Optional[float] = None

    warm_up_round: Optional[int] = None
    warm_up_strategy: Optional[str] = None

    candidate_hypotheses: List[str] = field(default_factory=list)
    controlled_candidates: Optional[List[str]] = None

    # Expected entropy of the hypothesis generator (U^PH_t). Optional.
    generator_entropy: Optional[float] = None
    pool_size: Optional[int] = None

    selected_principle_id: Optional[str] = None
    selected_hypothesis: Optional[str] = None
    selection_method: str = "information_directed"  # "uncertainty_sampling", "random", "information_directed", etc.

    # For IDS diagnostics: hypothesis -> {"regret": float, "info_gain": float, "ratio": float}
    information_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)

    guidance_message: str = ""

    # ---- NEW / ADDITIONAL FIELDS for faithful metric computation ----
    # selected_hypothesis_expected_rewards: mapping principle_id -> expected reward r(h_t, P)
    #   This is the expectation of reward for the selected hypothesis under each principle
    #   according to the agent's models (not necessarily the observed environment).
    selected_hypothesis_expected_rewards: Optional[Dict[str, float]] = None

    # principle_optimal_values: mapping principle_id -> v(P) = max_h r(h, P)
    #   The agent can compute these by evaluating its hypothesis set under each candidate principle.
    principle_optimal_values: Optional[Dict[str, float]] = None

    # optional scalar information gain observed or estimated for the selected hypothesis (I_t)
    selected_information_gain: Optional[float] = None

    # Hypothesis Embedding for tracking the dynamics of the evolving
    hypothesis_embedding: List[float] = None


@dataclass
class TheoreticalMetrics:
    """
    Core theoretical metrics from the dual uncertainty minimization framework.
    Fields may be None if not computable from available logs for a round.
    """

    round_number: int

    # === 1. Core Uncertainty Metrics (Section 2.2) ===
    u_ep: Optional[float]  # Evidence-to-Principle Uncertainty: U^EP_t
    u_ph: Optional[float]  # Principle-to-Hypothesis Uncertainty: U^PH_t

    # Posterior probability of true principle (if true_principle_id is provided)
    true_principle_posterior: Optional[float]

    # === 2. Performance and Regret Metrics ===
    instantaneous_regret: Optional[
        float
    ]  # Δ_t = v* - E[r(h_t, P^*)] (requires v* / true principle)
    identification_regret: Optional[float]  # Δ_t^(ID) = v* - E_{P~p_t}[ v(P) ]
    ph_regret: Optional[float]  # Δ_t^(PH) = E_{P~p_t}[ v(P) - r(h_t,P) ]

    cumulative_regret: Optional[float]
    average_regret: Optional[float]

    # === 3. Principle Discovery Dynamics ===
    principle_set_size: Optional[int]
    anomaly_count: Optional[int]
    anomaly_threshold: Optional[float]
    new_principles_discovered: Optional[int]

    # === 4. Information and Generator Metrics ===
    information_gain: Optional[float]  # I_t for selected hypothesis
    cumulative_information: Optional[float]  # aggregate
    information_ratio: Optional[float]  # Γ(h_t)

    # === 5. Convergence and Learning Diagnostics ===
    max_principle_belief: Optional[float]
    principle_belief_entropy: Optional[float]
    belief_concentration_ratio: Optional[float]  # max_belief / (1/|P_t|)
    ids_criterion: Optional[float]


class FlowTracker:
    """
    Main tracking class for the PiEvo framework.

    Responsibilities:
      - Keep lists of the three log classes (principle/hypothesis/experiment).
      - Provide methods to append new logs (these are the only methods PiEvo must call).
      - Compute TheoreticalMetrics for any round (or for all rounds) from the logs.
      - Persist metrics and logs to disk (simple JSON dumps or pickles can be added later).
    """

    def __init__(self, target_dir: str = "evo_track_log", load_from_file: str = ""):

        self.principle_guidance_logs: List[PrincipleLog] = []
        self.hypothesis_guidance_logs: List[HypothesisLog] = []
        self.experiment_guidance_logs: List[ExperimentLog] = []
        self.theoretical_metrics_logs: List[TheoreticalMetrics] = []
        self._cumulative_regret: float = 0.0
        self._cumulative_information: float = 0.0
        self.log_dir = os.path.join(target_dir, "metrics")

        if load_from_file:
            self.is_load_from_file = True
            with open(load_from_file, "r") as f:
                logs = json.load(f)
                self.principle_guidance_logs = [
                    PrincipleLog(**log_dict)
                    for log_dict in logs.get("principle_logs", [])
                ]
                self.hypothesis_guidance_logs = [
                    HypothesisLog(**log_dict)
                    for log_dict in logs.get("hypothesis_logs", [])
                ]
                self.experiment_guidance_logs = [
                    ExperimentLog(**log_dict)
                    for log_dict in logs.get("experiment_logs", [])
                ]
                self.theoretical_metrics_logs = [
                    TheoreticalMetrics(**log_dict)
                    for log_dict in logs.get("theoretical_metrics", [])
                ]
                logger.warning(
                    f"Loaded from `{load_from_file}` successfully. Totally {len(self.principle_guidance_logs)} detected. "
                )

        else:
            self.is_load_from_file = False
            os.makedirs(self.log_dir, exist_ok=True)

    # -----------------------------
    # Recording methods (PiEvo calls these)
    # -----------------------------
    def record_principle_log(self, log: PrincipleLog) -> None:
        """Append a PrincipleLog for a round."""
        self.principle_guidance_logs.append(log)

    def record_hypothesis_log(self, log: HypothesisLog) -> None:
        """Append a HypothesisLog for a round."""
        self.hypothesis_guidance_logs.append(log)

    def record_experiment_log(self, log: ExperimentLog) -> None:
        """Append an ExperimentLog for a round."""
        self.experiment_guidance_logs.append(log)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    @staticmethod
    def _safe_mean_weighted(
        values: Dict[str, float], weights: Dict[str, float]
    ) -> Optional[float]:
        """
        Compute weighted average of `values` keyed by same keys as `weights`.
        Returns None if keys mismatch or denom is zero.
        """
        if values is None or weights is None:
            return None
        # intersection of keys
        keys = set(values.keys()) & set(weights.keys())
        if not keys:
            return None
        num = 0.0
        denom = 0.0
        for k in keys:
            w = weights[k]
            try:
                v = float(values[k])
            except Exception:
                return None
            num += w * v
            denom += w
        if denom == 0.0:
            return None
        return num / denom

    @staticmethod
    def _entropy_from_beliefs(beliefs: Dict[str, float]) -> Optional[float]:
        """Compute Shannon entropy given a discrete distribution dict. Return None if invalid."""
        if beliefs is None or len(beliefs) == 0:
            return None
        # normalize in case tiny numerical drift
        total = sum(beliefs.values())
        if total <= 0:
            return None
        ent = 0.0
        for p in beliefs.values():
            if p <= 0:
                continue
            prob = p / total
            ent -= prob * math.log(prob, 2)
        return ent

    def _get_empirical_metrics(self):
        """
        Helper to recalculate metrics based on the Empirical Best (Max Observed Reward).
        This fixes issues where runtime v* estimates were too low, causing negative regret.
        """
        import numpy as np

        rounds = []
        rewards = []
        info_gains = []

        # Extract raw data from experiment logs
        for e in self.experiment_guidance_logs:
            # We need valid rewards for regret calculation
            if e.observed_reward is not None:
                r = e.round_number
                rew = float(e.observed_reward)
                # Use saved info gain, default to epsilon to avoid div/0
                ig = (
                    e.selected_information_gain
                    if e.selected_information_gain is not None
                    else 1e-9
                )

                rounds.append(r)
                rewards.append(rew)
                info_gains.append(ig)

        if not rounds:
            return None

        rounds = np.array(rounds)
        rewards = np.array(rewards)
        info_gains = np.array(info_gains)

        # 1. Determine Empirical Optimal Value (Best seen so far globally)
        v_star_empirical = np.max(rewards)

        # 2. Recalculate Instantaneous Regret (Always >= 0)
        inst_regret = v_star_empirical - rewards
        # Clip to 0 just in case of float precision issues
        inst_regret = np.maximum(inst_regret, 0.0)

        # 3. Recalculate Cumulative Metrics
        cum_regret = np.cumsum(inst_regret)
        cum_info = np.cumsum(info_gains)

        # 4. Recalculate IDS Ratio (Delta^2 / I_t)
        # Add epsilon to info_gain to handle cases where I_t is 0 (pure exploitation)
        ids_ratios = (inst_regret**2) / (info_gains + 1e-9)

        return {
            "rounds": rounds,
            "inst_regret": inst_regret,
            "cum_regret": cum_regret,
            "cum_info": cum_info,
            "ids_ratios": ids_ratios,
            "v_star": v_star_empirical,
        }

    # -----------------------------
    # Metric computation
    # -----------------------------
    def compute_metrics_for_round(
        self,
        round_number: int,
        true_principle_id: Optional[str] = None,
        true_principle_vstar: Optional[float] = None,
    ) -> TheoreticalMetrics:
        """
        Compute TheoreticalMetrics for `round_number`.

        Inputs:
          - round_number: int (1-indexed rounds as in your logs)
          - true_principle_id: optional id of the true principle if known (simulation)
          - true_principle_vstar: optional v* value (if available directly). If not
            provided but principle_optimal_values exists in the ExperimentLog for the round
            and true_principle_id is provided, v* will be taken from there.

        Returns:
          - TheoreticalMetrics (fields set to None when not computable).
        """
        # find logs for this round
        p_log = next(
            (l for l in self.principle_guidance_logs if l.round_number == round_number),
            None,
        )
        h_log = next(
            (
                l
                for l in self.hypothesis_guidance_logs
                if l.round_number == round_number
            ),
            None,
        )
        e_log = next(
            (
                l
                for l in self.experiment_guidance_logs
                if l.round_number == round_number
            ),
            None,
        )

        # initialize metrics container with defaults
        metrics = TheoreticalMetrics(
            round_number=round_number,
            u_ep=None,
            u_ph=None,
            true_principle_posterior=None,
            instantaneous_regret=None,
            identification_regret=None,
            ph_regret=None,
            cumulative_regret=None,
            average_regret=None,
            principle_set_size=None,
            anomaly_count=None,
            anomaly_threshold=None,
            new_principles_discovered=None,
            information_gain=None,
            cumulative_information=None,
            information_ratio=None,
            max_principle_belief=None,
            principle_belief_entropy=None,
            belief_concentration_ratio=None,
            ids_criterion=None,
        )

        # --- Uncertainty metrics (U^EP, U^PH) ---
        if p_log is not None:
            metrics.u_ep = float(p_log.principle_uncertainty_uep)
            metrics.principle_set_size = int(p_log.num_principles)
            metrics.anomaly_count = int(p_log.num_anomalies)
            metrics.anomaly_threshold = float(p_log.anomaly_threshold)
            # beliefs:
            beliefs = p_log.principle_beliefs
        elif h_log is not None:
            metrics.u_ep = float(h_log.principle_uncertainty_uep)
            beliefs = h_log.principle_beliefs
        elif e_log is not None and e_log.principle_optimal_values is not None:
            # fallback: try to infer beliefs from experiment log if it included them (unlikely)
            beliefs = None
        else:
            beliefs = None

        # compute U^PH from experiment generator_entropy if available
        if e_log is not None:
            metrics.u_ph = e_log.generator_entropy

        # Belief-derived quantities
        if beliefs is not None:
            # max belief
            try:
                max_bel = max(beliefs.values()) if beliefs else None
            except Exception:
                max_bel = None
            metrics.max_principle_belief = (
                float(max_bel) if max_bel is not None else None
            )
            metrics.principle_belief_entropy = self._entropy_from_beliefs(beliefs)
            if metrics.principle_belief_entropy is None and metrics.u_ep is not None:
                # if entropy given by p_log, keep that
                metrics.principle_belief_entropy = metrics.u_ep
            if (
                metrics.max_principle_belief is not None
                and metrics.principle_set_size is not None
                and metrics.principle_set_size > 0
            ):
                uniform_belief = 1.0 / metrics.principle_set_size
                metrics.belief_concentration_ratio = (
                    metrics.max_principle_belief / uniform_belief
                )
        else:
            # if p_log provided but beliefs missing (unlikely), use its UEP and leave belief fields None
            pass

        # --- Information gain and IDS metrics ---
        if e_log is not None:
            # Convert selected_hypothesis to a hashable key for lookup in information_ratios
            if isinstance(e_log.selected_hypothesis, dict):
                hyp_key = json.dumps(e_log.selected_hypothesis, sort_keys=True)
            else:
                hyp_key = e_log.selected_hypothesis

            # selection-specific info gain: prefer explicit field, otherwise read from information_ratios if present
            metrics.information_gain = e_log.selected_information_gain
            if (
                metrics.information_gain is None
                and e_log.information_ratios
                and hyp_key in e_log.information_ratios
            ):
                metrics.information_gain = e_log.information_ratios[hyp_key].get(
                    "info_gain"
                )
            # information ratio (IDS) for chosen hypothesis:
            if e_log.information_ratios and hyp_key in e_log.information_ratios:
                metrics.information_ratio = e_log.information_ratios[hyp_key].get(
                    "ratio"
                )

        # accumulate cumulative information
        if metrics.information_gain is not None:
            self._cumulative_information += metrics.information_gain
        metrics.cumulative_information = self._cumulative_information

        # --- Regret and decomposition ---
        # We use the decomposition:
        # Δ_t = v* - E_{P~p_t}[ r(h_t, P) ]
        #     = (v* - E_{P~p_t}[ v(P) ]) + (E_{P~p_t}[ v(P) - r(h_t,P) ])
        #
        # Required pieces:
        #  - v*: either true_principle_vstar (if provided) or taken from e_log.principle_optimal_values[true_principle_id]
        #  - beliefs: posterior over principles (p_t)
        #  - principle_optimal_values: mapping principle_id -> v(P)
        #  - selected_hypothesis_expected_rewards: mapping principle_id -> r(h_t, P)
        #
        v_star = None
        if true_principle_vstar is not None:
            v_star = float(true_principle_vstar)
        elif (
            true_principle_id is not None
            and e_log is not None
            and e_log.principle_optimal_values is not None
        ):
            v_star = float(e_log.principle_optimal_values.get(true_principle_id))

        # compute posterior prob of true principle if available
        if true_principle_id is not None and beliefs is not None:
            metrics.true_principle_posterior = float(
                beliefs.get(true_principle_id, 0.0)
            )

        # compute E_{P~p_t}[ v(P) ] if possible
        Evp = None
        if (
            e_log is not None
            and e_log.principle_optimal_values is not None
            and beliefs is not None
        ):
            Evp = self._safe_mean_weighted(e_log.principle_optimal_values, beliefs)

        # compute E_{P~p_t}[ r(h_t,P) ] if possible
        Erhp = None
        if (
            e_log is not None
            and e_log.selected_hypothesis_expected_rewards is not None
            and beliefs is not None
        ):
            Erhp = self._safe_mean_weighted(
                e_log.selected_hypothesis_expected_rewards, beliefs
            )

        # instantaneous regret: if v* known and Erhp known
        if v_star is not None and Erhp is not None:
            metrics.instantaneous_regret = float(v_star - Erhp)

        # identification_regret: v* - E_{P~p_t}[ v(P) ]
        if v_star is not None and Evp is not None:
            metrics.identification_regret = float(v_star - Evp)

        # ph_regret: E_{P~p_t}[ v(P) - r(h_t,P) ] = Evp - Erhp
        if Evp is not None and Erhp is not None:
            metrics.ph_regret = float(Evp - Erhp)

        # cumulative regret update: if instantaneous_regret computed, update running cumulative
        if metrics.instantaneous_regret is not None:
            # ensure non-negative numeric
            self._cumulative_regret += float(metrics.instantaneous_regret)
        metrics.cumulative_regret = float(self._cumulative_regret)
        # average regret = cumulative / t
        try:
            metrics.average_regret = metrics.cumulative_regret / float(round_number)
        except Exception:
            metrics.average_regret = None

        # --- Additional diagnostics from logs ---
        if p_log is not None:
            metrics.new_principles_discovered = (
                None  # not tracked in PrincipleLog; PiEvo may add if desired
            )
            # if PiEvo wants this tracked, it should include it in PrincipleLog and FlowTracker will pick it up
        if e_log is not None:
            metrics.ids_criterion = None
            # the IDS criterion for the chosen hypothesis can be computed from information_ratios if provided:
            if isinstance(e_log.selected_hypothesis, dict):
                ids_hyp_key = json.dumps(e_log.selected_hypothesis, sort_keys=True)
            else:
                ids_hyp_key = e_log.selected_hypothesis
            if e_log.information_ratios and ids_hyp_key in e_log.information_ratios:
                metrics.ids_criterion = e_log.information_ratios[ids_hyp_key].get(
                    "ratio"
                )

        # Save metrics to internal log
        self.theoretical_metrics_logs.append(metrics)
        return metrics

    def compute_all_metrics(
        self,
        true_principle_id: Optional[str] = None,
        true_principle_vstar: Optional[float] = None,
        absolute_value: Optional[float] = None,
    ) -> List[TheoreticalMetrics]:
        """
        Compute metrics for all rounds for which we have an ExperimentLog.
        If metrics for a round were previously computed, this will recompute and
        re-append (caller may want to clear previous metrics first if desired).
        """
        all_rounds = sorted({log.round_number for log in self.experiment_guidance_logs})
        metrics_list = []
        for r in all_rounds:
            metrics = self.compute_metrics_for_round(
                r, true_principle_id, true_principle_vstar
            )
            metrics_list.append(metrics)

        # ---- Visualization (Original) ----
        try:
            self.plot_uncertainty()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_uncertainty` failed due to the reason: {e}"
            )

        try:
            self.plot_regret()
        except Exception as e:
            logger.error(f"Plotting the `plot_regret` failed due to the reason: {e}")

        try:
            self.plot_principle_space()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_principle_space` failed due to the reason: {e}"
            )

        try:
            if absolute_value is not None:
                self.plot_experiment_quality(absolute_value)
        except Exception as e:
            logger.error(
                f"Plotting the `plot_experiment_quality` failed due to the reason: {e}"
            )

        try:
            self.plot_hypothesis_embedding(method="pca")
            self.plot_hypothesis_embedding(method="umap")
        except Exception as e:
            logger.error(
                f"Plotting the `plot_hypothesis_embedding` failed due to the reason: {e}"
            )

        try:
            self.plot_solution_quality_curve()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_solution_quality_curve` failed due to the reason: {e}"
            )

        try:
            self.plot_information_vs_regret()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_information_vs_regret` failed due to the reason: {e}"
            )

        try:
            self.plot_selection_method_effects()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_selection_method_effects` failed due to the reason: {e}"
            )

        try:
            self.plot_cumulative_regret_fit()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_cumulative_regret_fit` failed due to the reason: {e}"
            )

        try:
            self.plot_guidance_type_evolution()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_guidance_type_evolution` failed due to the reason: {e}"
            )

        try:
            self.plot_regret_decomposition()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_regret_decomposition` failed due to the reason: {e}"
            )

        try:
            self.plot_dual_uncertainty()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_dual_uncertainty` failed due to the reason: {e}"
            )

        try:
            self.plot_principle_dynamics()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_principle_dynamics` failed due to the reason: {e}"
            )

        try:
            if true_principle_id:
                self.plot_true_principle_belief()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_true_principle_belief` failed due to the reason: {e}"
            )

        try:
            self.plot_belief_heatmap(true_principle_id)
        except Exception as e:
            logger.error(
                f"Plotting the `plot_belief_heatmap` failed due to the reason: {e}"
            )

        try:
            self.plot_ids_candidate_space()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_ids_candidate_space` failed due to the reason: {e}"
            )

        try:
            self.plot_watershed_phenomenon()
        except Exception as e:
            logger.error(
                f"Plotting the `plot_watershed_phenomenon` failed due to the reason: {e}"
            )

        try:
            self.plot_dual_uncertainty_phase_space()
        except Exception as e:
            logger.error(f"Plotting `plot_dual_uncertainty_phase_space` failed: {e}")

        try:
            self.plot_principle_survival_stream()
        except Exception as e:
            logger.error(f"Plotting `plot_principle_survival_stream` failed: {e}")

        try:
            self.plot_learning_efficiency_frontier()
        except Exception as e:
            logger.error(f"Plotting `plot_learning_efficiency_frontier` failed: {e}")

        try:
            self.plot_ids_dynamics()
        except Exception as e:
            logger.error(f"Plotting `plot_ids_dynamics` failed: {e}")

        if not self.is_load_from_file:
            self.export_logs()
        return metrics_list

    # Plotting
    def _configure_plot_style(self):
        """Configure Seaborn & Matplotlib aesthetics for academic print style."""
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(
            style="white",
            context="paper",
            font="serif",
            rc={
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            },
        )
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["figure.dpi"] = 300

    def _save_figure(self, fig, name: str):
        """Helper to save figure as PDF and PNG in log_dir."""
        import os

        # Save PDF
        path_pdf = os.path.join(self.log_dir, f"{name}.pdf")
        fig.savefig(path_pdf, bbox_inches="tight", dpi=300)

        # Save PNG (for web dashboard)
        # path_png = os.path.join(self.log_dir, f"{name}.png")
        # fig.savefig(path_png, bbox_inches="tight", dpi=150)  # Lower DPI for web is fine

        logger.debug(f"Saved plot: {path_pdf} only. ")

    def plot_guidance_type_evolution(self):
        """
        Plots the evolution of `guidance_type` for Principles and Hypotheses
        using a "Gantt-like" horizontal bar chart (broken_barh).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from collections import defaultdict

        self._configure_plot_style()

        logs_map = {
            "Principle": self.principle_guidance_logs,
            "Hypothesis": self.hypothesis_guidance_logs,
        }

        if not logs_map["Principle"] and not logs_map["Hypothesis"]:
            logger.warning(
                "No principle or hypothesis logs to plot guidance type evolution."
            )
            return

        fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        fig.suptitle("Guidance Type Evolution", y=1.02)

        # Get a consistent color palette for all types across both plots
        all_types = set()
        for log_list in logs_map.values():
            for log in log_list:
                if log.guidance_type:
                    all_types.add(log.guidance_type)

        palette = sns.color_palette("pastel", n_colors=len(all_types) or 1)
        color_map = {
            dtype: color for dtype, color in zip(sorted(list(all_types)), palette)
        }

        def _get_segments(
            logs: List[Union[PrincipleLog, HypothesisLog]],
        ) -> (dict, list):
            """Helper to find contiguous segments of the same guidance type."""
            segments = defaultdict(list)
            all_types_in_log = []

            if not logs:
                return {}, []

            current_type = None
            start_round = -1

            for log in sorted(logs, key=lambda x: x.round_number):
                round_num = log.round_number
                g_type = log.guidance_type

                if g_type not in all_types_in_log:
                    all_types_in_log.append(g_type)

                if g_type != current_type:
                    # End the previous segment
                    if current_type is not None:
                        segments[current_type].append(
                            (start_round, round_num - start_round)
                        )
                    # Start a new segment
                    current_type = g_type
                    start_round = round_num

            # Add the final segment
            if current_type is not None:
                segments[current_type].append(
                    (start_round, logs[-1].round_number - start_round + 1)
                )

            return segments, all_types_in_log

        for ax, (name, logs) in zip(axes, logs_map.items()):
            segments, y_labels = _get_segments(logs)

            if not segments:
                ax.text(0.5, 0.5, f"No {name} logs", ha="center", va="center")
                ax.set_title(name)
                continue

            y_ticks = list(range(len(y_labels)))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

            for i, dtype in enumerate(y_labels):
                if dtype in segments:
                    ax.broken_barh(
                        segments[dtype],
                        (i - 0.4, 0.8),
                        facecolors=color_map.get(dtype, "gray"),
                    )

            ax.set_title(f"{name} Guidance")
            ax.set_ylabel("Guidance Type")
            ax.grid(axis="x", linestyle="--", alpha=0.6)

        axes[1].set_xlabel("Round")
        plt.tight_layout()
        self._save_figure(fig, "Guidance_type_evolution")
        plt.close(fig)

    def plot_uncertainty(self):
        """Plot U_EP evolution over rounds."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot U_EP.")
            return

        rounds = [m.round_number for m in self.theoretical_metrics_logs]
        u_ep = [m.u_ep for m in self.theoretical_metrics_logs]

        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        sns.lineplot(x=rounds, y=u_ep, ax=ax, marker="o", lw=1.2, color="#c43b64")
        ax.set_xlabel("Round")
        ax.set_ylabel(r"$U^{EP}$")
        ax.set_title("Evidence-to-Principle Uncertainty")
        self._save_figure(fig, "U_EP_vs_round")
        plt.close(fig)

    def plot_regret(self):
        """Plot instantaneous and average regret with scatter and fitted line."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.stats import linregress

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot regret.")
            return

        # Extract data
        rounds = np.array([m.round_number for m in self.theoretical_metrics_logs])
        inst_regret_raw = [
            m.instantaneous_regret for m in self.theoretical_metrics_logs
        ]
        avg_regret_raw = [m.average_regret for m in self.theoretical_metrics_logs]

        # Safely coerce to numeric and mask invalid values
        def _safe_to_float_list(vals):
            clean = []
            for v in vals:
                try:
                    clean.append(float(v))
                except (TypeError, ValueError):
                    clean.append(np.nan)
            return np.array(clean, dtype=float)

        inst_regret = _safe_to_float_list(inst_regret_raw)
        avg_regret = _safe_to_float_list(avg_regret_raw)

        # Filter only valid (finite) regret values for fitting
        valid_mask = np.isfinite(inst_regret)
        if not np.any(valid_mask):
            logger.warning("No finite regret values to plot.")
            return

        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        sns.scatterplot(
            x=rounds,
            y=inst_regret,
            ax=ax,
            label="Instantaneous Regret",
            color="firebrick",
            s=12,
        )
        sns.lineplot(
            x=rounds, y=avg_regret, ax=ax, label="Average Regret", color="navy", lw=1.2
        )

        # Optional fitted line for instantaneous regret
        if np.sum(valid_mask) > 2:
            slope, intercept, _, _, _ = linregress(
                rounds[valid_mask], inst_regret[valid_mask]
            )
            fit_y = intercept + slope * rounds
            sns.lineplot(
                x=rounds, y=fit_y, ax=ax, color="darkorange", lw=1, label="Linear Fit"
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Regret")
        ax.set_title("Instantaneous and Average Regret")
        ax.legend(frameon=True)
        self._save_figure(fig, "Regret_vs_round")
        plt.close(fig)

    def plot_principle_space(self):
        """Visualize principle space evolution using belief entropy and max belief."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot principle space.")
            return

        rounds = [m.round_number for m in self.theoretical_metrics_logs]
        entropy = [m.principle_belief_entropy for m in self.theoretical_metrics_logs]
        max_belief = [m.max_principle_belief for m in self.theoretical_metrics_logs]
        concentration = [
            m.belief_concentration_ratio for m in self.theoretical_metrics_logs
        ]

        fig, ax1 = plt.subplots(figsize=(3.2, 2.4))
        sns.lineplot(x=rounds, y=entropy, ax=ax1, color="teal", lw=1.2, label="Entropy")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Entropy", color="teal")

        ax2 = ax1.twinx()
        sns.lineplot(
            x=rounds,
            y=max_belief,
            ax=ax2,
            color="crimson",
            lw=1.0,
            linestyle="--",
            label="Max Belief",
        )
        ax2.set_ylabel("Max Belief", color="crimson")

        ax1.set_title("Principle Space Evolution")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)

        if ax2.get_legend():
            ax2.get_legend().remove()

        self._save_figure(fig, "Principle_space_evolution")
        plt.close(fig)

    def plot_experiment_quality(self, absolute_value: float):
        """
        Plot Solution Quality (SQ) and Area Under the Curve (AUC)
        given experiment outcomes and absolute reference mean.

        Args:
            absolute_value: μ_absolute for normalization.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        if not self.experiment_guidance_logs:
            logger.warning("No experiment logs to plot quality metrics.")
            return

        rounds_raw = [
            e.round_number
            for e in self.experiment_guidance_logs
            if e.observed_reward is not None
        ]
        rewards_raw = [
            e.observed_reward
            for e in self.experiment_guidance_logs
            if e.observed_reward is not None
        ]

        if len(rounds_raw) < 2 or absolute_value == 0:
            logger.warning("Insufficient data for SQ/AUC computation.")
            return

        rounds = np.array(rounds_raw)
        rewards = np.array(rewards_raw)

        # Compute SQ and AUC
        max_y = np.maximum.accumulate(rewards)
        SQ = (max_y / absolute_value) * 100.0

        # Calculate AUC correctly
        auc_values = (rewards[:-1] + rewards[1:]) / 2.0
        auc_cumulative = np.cumsum(auc_values)
        auc_rounds = np.arange(1, len(rounds))
        AUC = (auc_cumulative / (absolute_value * auc_rounds)) * 100.0

        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        sns.lineplot(
            x=rounds,
            y=SQ,
            ax=ax,
            label="Solution Quality (SQ)",
            color="#915484",
            lw=1.2,
        )
        sns.lineplot(
            x=rounds[1:],
            y=AUC,
            ax=ax,
            label="Area Under Curve (AUC)",
            color="#395eda",
            lw=1.2,
        )

        if len(SQ) > 0:
            ax.annotate(
                f"{SQ[-1]:.1f}%",
                (rounds[-1], SQ[-1]),
                textcoords="offset points",
                xytext=(5, 0),  # 5 points to the right
                ha="left",
                fontsize=8,
            )

        if len(AUC) > 0:
            ax.annotate(
                f"{AUC[-1]:.1f}%",
                (rounds[-1], AUC[-1]),  # Use last round for x-axis
                textcoords="offset points",
                xytext=(5, 0),  # 5 points to the right
                ha="left",
                fontsize=8,
                color="#395eda",
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Quality (%)")
        ax.set_title("Experiment Quality Metrics")
        ax.legend(frameon=True)
        self._save_figure(fig, "Experiment_quality")
        plt.close(fig)

    def plot_hypothesis_embedding(
        self, method: str = "pca", n_components: int = 2, random_state: int = 95
    ):
        """
        Visualize hypothesis embeddings across rounds using UMAP / t-SNE / PCA.
        - method: one of {"umap", "tsne", "pca"}; will fall back gracefully if library not available.
        - Uses each ExperimentLog.hypothesis_embedding (list[float]) as the vector for that round.
        - Colors points and trajectory by round_number.
        - Sizes points by information_gain (if available).
        - Includes a filled contour plot for the landscape based on observed_reward.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            from scipy.interpolate import griddata, RBFInterpolator
        except ImportError:
            logger.warning("scipy.interpolate not found. Skipping landscape fitting.")
            griddata = None
            RBFInterpolator = None

        self._configure_plot_style()

        # collect embeddings and metadata
        rounds = []
        embeddings = []
        info_gain = []
        observed_rewards = []  # NEW: Collect observed rewards for the landscape
        selection_method = []
        for e in self.experiment_guidance_logs:
            emb = getattr(e, "hypothesis_embedding", None)
            if emb is None:
                continue
            try:
                vec = np.asarray(emb, dtype=float)
            except Exception:
                continue
            embeddings.append(vec)
            rounds.append(e.round_number)
            info_gain.append(
                e.selected_information_gain
                if getattr(e, "selected_information_gain", None) is not None
                else float("nan")
            )
            observed_rewards.append(
                e.observed_reward
                if getattr(e, "observed_reward", None) is not None
                else float("nan")
            )
            selection_method.append(
                e.selection_method
                if getattr(e, "selection_method", None) is not None
                else "unknown"
            )

        if len(embeddings) < 3:  # Need at least 3 points for reduction/interpolation
            logger.warning(
                "Not enough hypothesis embeddings found in experiment logs for embedding plot."
            )
            return

        X = np.vstack(embeddings)
        # dimensionality reduction: try umap, then tsne, then pca
        embedding_2d = None
        method_used = None

        if method == "umap":
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=n_components, random_state=random_state
                )
                embedding_2d = reducer.fit_transform(X)
                method_used = "UMAP"
            except Exception:
                # fallback to TSNE
                method = "tsne"

        if method == "tsne" and embedding_2d is None:
            try:
                from sklearn.manifold import TSNE

                reducer = TSNE(
                    n_components=n_components,
                    random_state=random_state,
                    init="pca",
                    learning_rate="auto",
                )
                embedding_2d = reducer.fit_transform(X)
                method_used = "t-SNE"
            except Exception:
                method = "pca"

        if method == "pca" and embedding_2d is None:
            try:
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=n_components, random_state=random_state)
                embedding_2d = reducer.fit_transform(X)
                method_used = "PCA"
            except Exception:
                embedding_2d = None

        if embedding_2d is None:
            logger.warning(
                f"Failed to compute 2D embedding (method {method} failed or library not available)."
            )
            return

        # plotting
        fig, ax = plt.subplots(figsize=(4.0, 3.0))

        try:
            x_coords = embedding_2d[:, 0]
            y_coords = embedding_2d[:, 1]
            z_values = np.asarray(
                observed_rewards
            )  # MODIFIED: Use observed_reward for landscape

            # Filter out NaNs, griddata can't handle them
            valid_mask = (
                ~np.isnan(z_values) & np.isfinite(x_coords) & np.isfinite(y_coords)
            )
            x_filt = x_coords[valid_mask]
            y_filt = y_coords[valid_mask]
            z_filt = z_values[valid_mask]

            Z_grid = None  # Initialize Z_grid
            if len(x_filt) > 3:
                # 1. Create a grid to interpolate onto
                x_min, x_max = x_filt.min(), x_filt.max()
                y_min, y_max = y_filt.min(), y_filt.max()
                x_pad = (x_max - x_min) * 0.1
                y_pad = (y_max - y_min) * 0.1
                xi = np.linspace(x_min - x_pad, x_max + x_pad, 300)
                yi = np.linspace(y_min - y_pad, y_max + y_pad, 300)
                X_grid, Y_grid = np.meshgrid(xi, yi)

                if RBFInterpolator is not None:
                    try:
                        # 2. Interpolate z_values (reward) onto the grid
                        # 'thin_plate_spline' is excellent for smooth surfaces
                        rbf_interpolator = RBFInterpolator(
                            np.column_stack([x_filt, y_filt]),
                            z_filt,
                            kernel="thin_plate_spline",
                        )
                        Z_grid = rbf_interpolator(
                            np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                        )
                        Z_grid = Z_grid.reshape(X_grid.shape)
                    except Exception as rbf_e:
                        logger.warning(
                            f"RBFInterpolator failed ({rbf_e}), falling back to griddata."
                        )
                        Z_grid = None  # Ensure Z_grid is None to trigger fallback

                if Z_grid is None and griddata is not None:
                    # Fallback to cubic griddata if RBF failed or wasn't available
                    if (
                        RBFInterpolator is not None
                    ):  # Only log if we are in the fallback case
                        logger.warning(
                            "RBFInterpolator failed, falling back to griddata (may be sharp)."
                        )
                    Z_grid = griddata(
                        (x_filt, y_filt), z_filt, (X_grid, Y_grid), method="cubic"
                    )
                elif Z_grid is None:
                    # Both failed
                    logger.warning(
                        "No scipy interpolation tools (RBFInterpolator or griddata) found."
                    )

                if Z_grid is not None:
                    # 3. Plot the filled contour (the landscape)
                    contour = ax.contourf(
                        X_grid,
                        Y_grid,
                        Z_grid,
                        levels=6,  # Smooth transitions
                        cmap="magma",  # Changed to magma to match points
                        alpha=0.6,  # Slightly more transparent
                        zorder=1,  # Keep it in the back
                        antialiased=True,  # Smoother rendering
                    )

                    # 4. Add a colorbar for the landscape (reward)
                    # --- MODIFICATION: Moved to right side ---
                    cbar_reward = fig.colorbar(
                        contour, ax=ax, shrink=0.8, location="right", pad=0.02
                    )
                    cbar_reward.set_label("Observed Reward")

                    # 5. Set axis limits to match the interpolated grid
                    ax.set_xlim(xi.min(), xi.max())
                    ax.set_ylim(yi.min(), yi.max())
                else:
                    if griddata is None and RBFInterpolator is None:
                        pass  # Already warned at import
                    else:
                        logger.warning("Could not generate landscape grid.")
            else:
                logger.warning(
                    "Not enough valid observed_reward points to draw reward landscape."
                )
        except Exception as e:
            logger.warning(
                f"Could not draw reward landscape (griddata/contourf) for embedding plot: {e}"
            )  # MODIFIED: Log message

        # Robust sizing based on info_gain
        info_gain_arr = np.nan_to_num(np.asarray(info_gain), nan=0.0)
        if info_gain_arr.size > 0 and np.nanmax(info_gain_arr) != np.nanmin(
            info_gain_arr
        ):
            sizes = np.clip((info_gain_arr - np.nanmin(info_gain_arr)) + 20, 10, 200)
        else:
            sizes = 30  # Default size if no variance or empty

        # scatter points (colored by round)
        sc = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=rounds,
            cmap="magma",
            s=sizes,  # Size by info_gain
            alpha=0.95,
            edgecolors="none",
            zorder=3,  # Ensure points are on top of landscape
        )

        # Normalize colors based on "rounds" for the trajectory line
        norm = plt.Normalize(np.min(rounds), np.max(rounds))
        cmap = plt.get_cmap("magma")

        # Create line segments between consecutive points (the trajectory)
        points = embedding_2d.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with gradient coloring (also by round)
        lc = mcoll.LineCollection(
            segments,
            cmap=None,
            norm=norm,
            array=rounds[:-1],  # color along the line
            linewidth=1.5,
            alpha=0.8,
            zorder=2,  # Ensure line is on top of landscape
        )
        ax.add_collection(lc)

        ax.set_xlabel(f"{method_used} dim 1")
        ax.set_ylabel(f"{method_used} dim 2")
        ax.set_title(f"Hypothesis Embedding Evolution ({method_used})")

        # annotate key points
        try:
            idx_first = int(np.argmin(rounds))
            idx_last = int(np.argmax(rounds))
            idx_med = int(len(rounds) // 2)
            ax.annotate(
                "first",
                (embedding_2d[idx_first, 0], embedding_2d[idx_first, 1]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=8,
                zorder=4,
            )
            ax.annotate(
                "last",
                (embedding_2d[idx_last, 0], embedding_2d[idx_last, 1]),
                textcoords="offset points",
                xytext=(3, -10),
                fontsize=8,
                zorder=4,
            )
            ax.annotate(
                "mid",
                (embedding_2d[idx_med, 0], embedding_2d[idx_med, 1]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=8,
                zorder=4,
            )
        except Exception:
            pass

        self._save_figure(fig, f"Hypothesis_embedding_{method_used}")
        plt.close(fig)

    def plot_solution_quality_curve(self):
        """
        Plot solution quality (max-so-far) vs round and the reward trace.
        Reuses Plot aesthetics to match existing figures.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._configure_plot_style()

        rounds = [
            e.round_number
            for e in self.experiment_guidance_logs
            if e.observed_reward is not None
        ]
        rewards = [
            e.observed_reward
            for e in self.experiment_guidance_logs
            if e.observed_reward is not None
        ]

        if len(rewards) < 1:
            logger.warning("No observed rewards to plot solution quality.")
            return

        rounds = np.array(rounds)
        rewards = np.array(rewards, dtype=float)

        max_so_far = np.maximum.accumulate(rewards)
        improvement = max_so_far - np.concatenate(([0.0], max_so_far[:-1]))

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(4.0, 4.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 0.8]},
        )
        ax = axs[0]
        sns.lineplot(
            x=rounds,
            y=rewards,
            ax=ax,
            marker="o",
            label="Reward",
            lw=1.0,
            color="#926ca9",
        )
        sns.lineplot(
            x=rounds,
            y=max_so_far,
            ax=ax,
            marker="s",
            label="Max so far",
            lw=1.2,
            color="#090813",
        )
        ax.set_ylabel("Reward")
        ax.set_title("Reward Trace and Solution Quality")
        ax.legend(frameon=True)

        ax2 = axs[1]
        ax2.bar(rounds, improvement, color="#373469", width=0.8, alpha=0.7)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Improvement")
        self._save_figure(fig, "Solution_quality_and_improvement")
        plt.close(fig)

    def plot_information_vs_regret(self):
        """
        Scatter plot of per-round information gain vs instantaneous regret with a fit.
        Useful to inspect the IDS trade-off visually.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import linregress

        self._configure_plot_style()

        info = np.array(
            [m.information_gain for m in self.theoretical_metrics_logs], dtype=float
        )
        regret = np.array(
            [m.instantaneous_regret for m in self.theoretical_metrics_logs], dtype=float
        )
        rounds = np.array(
            [m.round_number for m in self.theoretical_metrics_logs], dtype=int
        )

        mask = np.isfinite(info) & np.isfinite(regret)
        if not np.any(mask):
            logger.warning("No finite information/regret pairs to plot.")
            return

        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        sns.scatterplot(
            x=info[mask],
            y=regret[mask],
            hue=rounds[mask],
            palette="magma",
            ax=ax,
            legend=False,
            s=30,
        )
        # fit a robust linear regression if enough points
        try:
            slope, intercept, r_value, p_value, std_err = linregress(
                info[mask], regret[mask]
            )
            xs = np.linspace(np.min(info[mask]), np.max(info[mask]), 100)
            ax.plot(
                xs,
                intercept + slope * xs,
                linestyle="--",
                linewidth=1.0,
                label=f"fit (r={r_value:.2f})",
            )
            ax.legend(frameon=True)
        except Exception:
            pass

        ax.set_xlabel("Information gain (I_t)")
        ax.set_ylabel("Instantaneous regret Δ_t")
        ax.set_title("Information vs Regret (per round)")
        self._save_figure(fig, "Information_vs_Regret")
        plt.close(fig)

    def plot_selection_method_effects(self):
        """
        Compare different selection methods across information gain,
        instantaneous regret, and observed reward.

        Robust against missing or NaN data per metric.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        self._configure_plot_style()

        rows = []
        metrics_logs = (
            self.theoretical_metrics_logs
            if self.theoretical_metrics_logs
            else [None] * len(self.experiment_guidance_logs)
        )
        for e, m in zip(self.experiment_guidance_logs, metrics_logs):
            rows.append(
                {
                    "round": e.round_number,
                    "selection_method": getattr(e, "selection_method", "unknown"),
                    "info_gain": getattr(e, "selected_information_gain", np.nan),
                    "inst_regret": getattr(m, "instantaneous_regret", np.nan)
                    if m is not None
                    else np.nan,
                    "observed_reward": getattr(e, "observed_reward", np.nan),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No data to plot selection method effects.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

        def safe_boxplot(ax, y_col, title, ylabel):
            """Helper to handle missing/NaN data gracefully."""
            sub = df[["selection_method", y_col]].dropna()
            if sub.empty or sub[y_col].nunique() == 0:
                ax.text(0.5, 0.5, f"No valid {y_col} data", ha="center", va="center")
                ax.set_axis_off()
                return
            sns.boxplot(x="selection_method", y=y_col, data=sub, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            for label in ax.get_xticklabels():
                label.set_rotation(25)
                label.set_horizontalalignment("right")

        safe_boxplot(
            axes[0], "info_gain", "Information Gain by Selection Method", "I_t"
        )
        safe_boxplot(
            axes[1], "inst_regret", "Instantaneous Regret by Selection Method", "Δ_t"
        )
        safe_boxplot(
            axes[2], "observed_reward", "Observed Reward by Selection Method", "Reward"
        )

        fig.suptitle("Selection Method Diagnostics", fontsize=11)
        fig.tight_layout()
        self._save_figure(fig, "Selection_method_effects")
        plt.close(fig)

    def plot_regret_decomposition(self):
        """Plot the decomposition of instantaneous regret over time."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot regret decomposition.")
            return

        data = []
        for m in self.theoretical_metrics_logs:
            data.append(
                {
                    "round": m.round_number,
                    "Total Regret ($\Delta_t$)": m.instantaneous_regret,
                    "Identification Regret ($\Delta_t^{ID}$)": m.identification_regret,
                    "PH Regret ($\Delta_t^{PH}$)": m.ph_regret,
                }
            )

        df = pd.DataFrame(data).melt(
            id_vars="round", var_name="Regret Component", value_name="Value"
        )
        df = df.dropna()

        if df.empty:
            logger.warning("No finite regret decomposition data to plot.")
            return

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.lineplot(
            data=df,
            x="round",
            y="Value",
            hue="Regret Component",
            ax=ax,
            lw=1.2,
            style="Regret Component",
            markers=True,
            dashes=False,
        )

        ax.set_xlabel("Round")
        ax.set_ylabel("Regret")
        ax.set_title("Regret Decomposition Over Time")
        ax.legend(frameon=True, fontsize=8)
        self._save_figure(fig, "Regret_decomposition")
        plt.close(fig)

    def plot_dual_uncertainty(self):
        """Plot U_EP and U_PH evolution on a dual-axis chart.
        If U_PH data is absent, plots only U_EP."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot dual uncertainty.")
            return

        rounds = [m.round_number for m in self.theoretical_metrics_logs]
        u_ep = [m.u_ep for m in self.theoretical_metrics_logs]
        u_ph_raw = [m.u_ph for m in self.theoretical_metrics_logs]

        # Check if there is any valid u_ph data to plot
        u_ph = np.array([v if v is not None else np.nan for v in u_ph_raw], dtype=float)
        has_u_ph_data = not np.isnan(u_ph).all()

        fig, ax1 = plt.subplots(figsize=(3.2, 2.4))

        # Plot U_EP (Evidence-to-Principle)
        sns.lineplot(
            x=rounds,
            y=u_ep,
            ax=ax1,
            marker="o",
            lw=1.2,
            color="#c43b64",
            label=r"$U^{EP}$ (Principle)",
        )
        ax1.set_xlabel("Round")
        ax1.set_ylabel(r"$U^{EP}$ (bits)", color="#c43b64")

        lines, labels = ax1.get_legend_handles_labels()

        if has_u_ph_data:
            # Plot U_PH (Principle-to-Hypothesis)
            ax2 = ax1.twinx()
            sns.lineplot(
                x=rounds,
                y=u_ph,
                ax=ax2,
                marker="s",
                lw=1.2,
                color="#005a9c",
                label=r"$U^{PH}$ (Hypothesis)",
            )
            ax2.set_ylabel(r"$U^{PH}$ (bits)", color="#005a9c")
            ax1.set_title("Dual Uncertainty Evolution")

            # Manually create a combined legend
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
            if ax2.get_legend():
                ax2.get_legend().remove()
        else:
            logger.warning(
                "No valid U_PH (generator_entropy) data found. Plotting U_EP only."
            )
            ax1.set_title("Principle Uncertainty Evolution")
            ax1.legend(loc="best", frameon=True)

        self._save_figure(fig, "Dual_Uncertainty_vs_round")
        plt.close(fig)

    def plot_principle_dynamics(self):
        """
        Plot principle set size and anomaly count over time.

        FIXED: This version sources all data directly from self.principle_guidance_logs
        to resolve shape mismatches and uses a line plot for anomalies as requested.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator

        self._configure_plot_style()

        # --- FIX: Source all data from principle_guidance_logs ---
        if not self.principle_guidance_logs:
            logger.warning("No principle logs to plot principle dynamics.")
            return

        # Get rounds, set_size, and anomalies directly from the principle logs
        # This guarantees all lists have the same length.
        rounds = [log.round_number for log in self.principle_guidance_logs]
        set_size = [
            log.num_principles if log.num_principles is not None else 0
            for log in self.principle_guidance_logs
        ]
        anomalies = [
            log.num_anomalies if log.num_anomalies is not None else 0
            for log in self.principle_guidance_logs
        ]
        # --- END FIX ---

        if not rounds:
            logger.warning(
                "plot_principle_dynamics: No rounds found in principle_guidance_logs."
            )
            return

        if all(s == 0 for s in set_size) and all(a == 0 for a in anomalies):
            logger.warning(
                "plot_principle_dynamics: All 'num_principles' and 'num_anomalies' values are 0 or None."
            )

        fig, ax1 = plt.subplots(figsize=(3.2, 2.4))

        # Plot Principle Set Size
        sns.lineplot(
            x=rounds,
            y=set_size,
            ax=ax1,
            marker="o",
            lw=1.2,
            color="darkgreen",
            label="Principle Set Size",
        )
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Count (Principles)", color="darkgreen")

        # Plot Anomaly Count
        ax2 = ax1.twinx()

        # --- FIX: Use lineplot instead of bar plot for anomalies ---
        sns.lineplot(
            x=rounds,
            y=anomalies,
            ax=ax2,
            marker="X",
            linestyle="--",
            lw=1.0,
            color="gray",
            label="Anomalies",
        )
        # --- END FIX ---

        ax2.set_ylabel("Count (Anomalies)", color="gray")

        # Set integer ticks for y-axes
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Set minimum y-limit for count data
        current_ylim_ax1 = ax1.get_ylim()
        ax1.set_ylim(
            bottom=min(current_ylim_ax1[0], -0.1), top=max(1, current_ylim_ax1[1])
        )

        current_ylim_ax2 = ax2.get_ylim()
        ax2.set_ylim(
            bottom=min(current_ylim_ax2[0], -0.1), top=max(1, current_ylim_ax2[1])
        )

        ax1.set_title("Principle Discovery Dynamics")

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
        try:
            ax2.get_legend().remove()
        except AttributeError:
            pass

        self._save_figure(fig, "Principle_dynamics")
        plt.close(fig)

    def plot_watershed_phenomenon(self, true_principle_id: Optional[str] = None):
        """
        Plots the "Watershed Phenomenon" retrospectively.

        Fixes "Prior beliefs are empty" by searching for the first round
        where beliefs actually exist (Effective Prior).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        self._configure_plot_style()

        if not self.principle_guidance_logs:
            logger.warning("No principle logs to plot watershed phenomenon.")
            return

        # 1. Sort logs by round
        sorted_logs = sorted(self.principle_guidance_logs, key=lambda x: x.round_number)

        # 2. Find the "Effective Prior Log" (First log with non-empty beliefs)
        prior_log = None
        start_index = 0

        for i, log in enumerate(sorted_logs):
            if log.principle_beliefs and len(log.principle_beliefs) > 0:
                prior_log = log
                start_index = i
                break

        if prior_log is None:
            logger.warning(
                "Prior beliefs are empty in ALL logs. Cannot plot watershed."
            )
            return

        p0_beliefs = prior_log.principle_beliefs
        logger.info(
            f"Watershed: Effective Prior found at Round {prior_log.round_number}"
        )

        # 3. Determine Proxy for Truth (P*) - looking at the LAST valid log
        if not true_principle_id:
            # Find the last log with beliefs
            last_valid_log = None
            for log in reversed(sorted_logs):
                if log.principle_beliefs:
                    last_valid_log = log
                    break

            if last_valid_log:
                true_principle_id = max(
                    last_valid_log.principle_beliefs,
                    key=last_valid_log.principle_beliefs.get,
                )
                logger.info(f"Watershed: Auto-selected '{true_principle_id}' as P*.")
            else:
                logger.warning("No valid beliefs found in any log to determine winner.")
                return

        # 4. Identify Roles (P* vs P_f) based on the Effective Prior
        # Find the "Incumbent" (Highest belief at t_start)
        incumbent_id = max(p0_beliefs, key=p0_beliefs.get)
        incumbent_belief = p0_beliefs[incumbent_id]

        # Constants
        VIRTUAL_MASS = 1e-3  # Mass for principles that don't exist yet
        epsilon = 1e-12

        # Scenario A: The Winner is a New Principle (Revolution)
        if true_principle_id != incumbent_id:
            p_f_id = incumbent_id

            # P* might not exist in prior, use Virtual Mass if missing
            p0_star = p0_beliefs.get(true_principle_id, VIRTUAL_MASS)
            p0_false = incumbent_belief

            plot_title = "Watershed: New Principle Overcoming Incumbent"

        # Scenario B: The Winner is the Incumbent (Dominance)
        else:
            p_f_id = "Virtual_Competitor"
            p0_star = incumbent_belief
            p0_false = VIRTUAL_MASS
            plot_title = "Dominance: Incumbent Maintaining Lead"

        # 5. Calculate Bias and Evidence
        # RHS (Bias): log( p0(Pf) / p0(P*) )
        log_prior_bias_rhs = np.log2(p0_false + epsilon) - np.log2(p0_star + epsilon)

        # Base odds for normalization: log( p0(P*) / p0(Pf) )
        log_prior_odds_base = np.log2(p0_star + epsilon) - np.log2(p0_false + epsilon)

        plot_data = []

        # Iterate starting from the Effective Prior Log
        for log in sorted_logs[start_index:]:
            pT_beliefs = log.principle_beliefs
            if not pT_beliefs:
                continue

            # Get Belief of P*
            pT_star = pT_beliefs.get(true_principle_id, epsilon)

            # Get Belief of P_f
            if p_f_id == "Virtual_Competitor":
                # Competitor is "Rest of World"
                pT_false = max(1.0 - pT_star, epsilon)
            else:
                pT_false = pT_beliefs.get(p_f_id, epsilon)

            # Avoid log(0)
            pT_star = max(pT_star, epsilon)
            pT_false = max(pT_false, epsilon)

            # log2( p_T(P*) / p_T(P_f) )
            log_posterior_odds = np.log2(pT_star) - np.log2(pT_false)

            # LHS: Cumulative Evidence
            log_cumulative_evidence_lhs = log_posterior_odds - log_prior_odds_base

            plot_data.append(
                {
                    "round": log.round_number,
                    "LHS (Evidence)": log_cumulative_evidence_lhs,
                    "RHS (Bias)": log_prior_bias_rhs,
                }
            )

        if not plot_data:
            return

        df = pd.DataFrame(plot_data)

        # 6. Plot
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        sns.lineplot(
            data=df,
            x="round",
            y="RHS (Bias)",
            ax=ax,
            lw=1.5,
            linestyle="--",
            color="red",
            label="Prior Bias",
        )
        sns.lineplot(
            data=df,
            x="round",
            y="LHS (Evidence)",
            ax=ax,
            lw=1.5,
            color="blue",
            label="Cum. Evidence",
        )

        # Find Watershed
        if log_prior_bias_rhs > 0:
            try:
                watershed_df = df[df["LHS (Evidence)"] > df["RHS (Bias)"]]
                if not watershed_df.empty:
                    w_round = watershed_df["round"].min()
                    ax.axvline(
                        x=w_round, color="green", linestyle=":", label="Watershed"
                    )
                    ax.plot(w_round, df["RHS (Bias)"].iloc[0], "go", markersize=5)
            except Exception:
                pass

        ax.set_xlabel("Round")
        ax.set_ylabel("Log-Odds (bits)")

        # Safe title handling
        # short_true = true_principle_id[:15] + "..." if len(true_principle_id) > 20 else true_principle_id
        ax.set_title(f"{plot_title}\n($P^*=${true_principle_id})", fontsize=9)
        ax.legend(frameon=True, fontsize=7)

        self._save_figure(fig, "Watershed_Phenomenon")
        plt.close(fig)

    def plot_cumulative_regret_fit(self):
        """
        Plot cumulative regret and fit sub-linear functions (sqrt(t) and log(t)).

        NOTE: This function *re-computes* the cumulative regret based on observed
        rewards. It defines v* as the *best observed reward* across all rounds
        (empirical v*), which is a common practice when the true v* is unknown.
        This avoids issues where the pre-computed regret is zero.

        --- NEW: Now includes R-squared (R²) goodness-of-fit in the legend. ---
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        try:
            from scipy.optimize import curve_fit
        except ImportError:
            logger.warning(
                "scipy.optimize not found. Skipping cumulative regret fitting."
            )
            return

        self._configure_plot_style()

        # --- MODIFICATION: Re-calculate regret from observed rewards ---

        # 1. Get all observed rewards and their rounds
        if not self.experiment_guidance_logs:
            logger.warning("No experiment logs found to calculate empirical regret.")
            return

        rounds_rewards = []
        for e in self.experiment_guidance_logs:
            # Only use rounds that have a valid, non-None observed_reward
            if e.observed_reward is not None:
                try:
                    rounds_rewards.append((e.round_number, float(e.observed_reward)))
                except (ValueError, TypeError):
                    continue  # Skip if reward is not a number

        if not rounds_rewards:
            logger.warning(
                "No valid 'observed_reward' data found in experiment logs for regret plot."
            )
            return

        # Sort by round number to ensure correct cumulative sum
        rounds_rewards.sort(key=lambda x: x[0])

        rounds_raw = np.array([r for r, rew in rounds_rewards], dtype=float)
        rewards_raw = np.array([rew for r, rew in rounds_rewards], dtype=float)

        # 2. Find the empirical v* (best-found reward)
        v_star_empirical = np.max(rewards_raw)

        # 3. Calculate instantaneous regret for each round
        # inst_regret = v_star_empirical - reward_at_t
        inst_regret = v_star_empirical - rewards_raw

        # 4. Calculate cumulative regret
        cum_regret_recalc = np.cumsum(inst_regret)

        # 5. Assign to t and regret (replacing the old data)
        t = rounds_raw
        regret = cum_regret_recalc

        if len(t) < 3:  # Need at least 3 points to fit
            logger.warning(
                "Not enough valid cumulative regret data points (< 3) to plot or fit."
            )
            return
        # --- END MODIFICATION ---

        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        # Plot actual cumulative regret (now correctly computed)
        sns.lineplot(
            x=t,
            y=regret,
            ax=ax,
            lw=1.5,
            color="firebrick",
            label="Cumulative Regret $R_T$",
        )

        # --- NEW: Calculate Total Sum of Squares (SS_tot) for R² ---
        # R² = 1 - (SS_res / SS_tot)
        SS_tot = np.sum((regret - np.mean(regret)) ** 2)
        # Add epsilon to avoid division by zero if data is constant
        if SS_tot == 0:
            SS_tot = 1e-9
            # --- END NEW ---

        # Define fit functions
        def func_sqrt(t, a, b):
            return a * np.sqrt(t) + b

        def func_log(t, a, b):
            # Add small epsilon to avoid log(0) if round 0 is present
            return a * np.log(t + 1e-9) + b

        # Fit sqrt(t)
        try:
            popt_sqrt, _ = curve_fit(
                func_sqrt, t, regret, p0=[1, 0]
            )  # Add initial guess
            a_sqrt, b_sqrt = popt_sqrt

            # --- NEW: Calculate R² for sqrt fit ---
            y_pred_sqrt = func_sqrt(t, a_sqrt, b_sqrt)
            SS_res_sqrt = np.sum((regret - y_pred_sqrt) ** 2)
            r2_sqrt = 1 - (SS_res_sqrt / SS_tot)
            # --- END NEW ---

            sns.lineplot(
                x=t,
                y=y_pred_sqrt,
                ax=ax,
                lw=1.0,
                linestyle="--",
                # --- MODIFIED: Added R² to label ---
                label=f"${a_sqrt:.2f} \cdot \sqrt{{t}} + {b_sqrt:.2f} \ (R^2={r2_sqrt:.2f})$",
            )
        except Exception as e:
            logger.warning(f"Failed to fit sqrt(t) to regret: {e}")

        # Fit log(t)
        try:
            popt_log, _ = curve_fit(func_log, t, regret, p0=[1, 0])  # Add initial guess
            a_log, b_log = popt_log

            # --- NEW: Calculate R² for log fit ---
            y_pred_log = func_log(t, a_log, b_log)
            SS_res_log = np.sum((regret - y_pred_log) ** 2)
            r2_log = 1 - (SS_res_log / SS_tot)
            # --- END NEW ---

            sns.lineplot(
                x=t,
                y=y_pred_log,
                ax=ax,
                lw=1.0,
                linestyle=":",
                # --- MODIFIED: Added R² to label ---
                label=f"${a_log:.2f} \cdot \log(t) + {b_log:.2f} \ (R^2={r2_log:.2f})$",
            )
        except Exception as e:
            logger.warning(f"Failed to fit log(t) to regret: {e}")

        ax.set_xlabel("Round (T)")
        ax.set_ylabel("Cumulative Regret ($R_T$)")
        ax.set_title("Cumulative Regret and Sub-linear Fit")
        ax.legend(frameon=True, fontsize=8)  # Fontsize reduced to fit new labels

        # Set y-axis to start at or below zero
        ax.set_ylim(
            bottom=min(0, np.min(regret))
            - 0.05 * (np.max(regret) - np.min(regret) + 1e-6)
        )

        self._save_figure(fig, "Cumulative_Regret_Fit")
        plt.close(fig)

    def plot_true_principle_belief(self):
        """Plot the posterior probability of the true principle over time."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics computed to plot true principle belief.")
            return

        rounds = np.array([m.round_number for m in self.theoretical_metrics_logs])
        beliefs = np.array(
            [m.true_principle_posterior for m in self.theoretical_metrics_logs],
            dtype=float,
        )

        if np.isnan(beliefs).all():
            logger.warning(
                "No true_principle_posterior data available (was true_principle_id provided?)."
            )
            return

        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        sns.lineplot(x=rounds, y=beliefs, ax=ax, marker=".", lw=1.2, color="purple")
        ax.set_xlabel("Round")
        ax.set_ylabel(r"$p_t(P^*)$")
        ax.set_title("Belief in True Principle")
        ax.set_ylim(-0.05, 1.05)
        self._save_figure(fig, "True_principle_belief")
        plt.close(fig)

    def plot_belief_heatmap(self, true_principle_id: Optional[str] = None):
        """Plot a heatmap of principle beliefs over time."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        self._configure_plot_style()

        if not self.principle_guidance_logs:
            logger.warning("No principle logs to plot belief heatmap.")
            return

        # Collect all belief data
        data = []
        all_principles = []
        for log in self.principle_guidance_logs:
            if log.principle_beliefs:
                for p_id, belief in log.principle_beliefs.items():
                    data.append(
                        {
                            "round": log.round_number,
                            "principle_id": p_id,
                            "belief": belief,
                        }
                    )
                    if p_id not in all_principles:
                        all_principles.append(p_id)

        if not data:
            logger.warning("No belief data found in principle logs.")
            return

        df = pd.DataFrame(data)

        # Pivot to create [principles x rounds] matrix
        belief_matrix = df.pivot(index="principle_id", columns="round", values="belief")
        # Re-order index to be in order of appearance
        belief_matrix = belief_matrix.reindex(all_principles).fillna(0.0)

        fig, ax = plt.subplots(
            figsize=(
                max(6, len(belief_matrix.columns) * 0.3),
                max(4, len(belief_matrix.index) * 0.3),
            )
        )
        sns.heatmap(
            belief_matrix,
            ax=ax,
            cmap="magma",
            annot=False,
            linewidths=0.5,
            cbar_kws={"label": "Belief $p_t(P)$"},
        )

        # Highlight true principle
        if true_principle_id in belief_matrix.index:
            try:
                # Find the integer index of the true principle
                true_idx = belief_matrix.index.get_loc(true_principle_id)
                ax.get_yticklabels()[true_idx].set_weight("bold")
                ax.get_yticklabels()[true_idx].set_color("red")
            except Exception:
                pass  # fail silently if highlighting fails

        ax.set_title("Principle Belief Evolution")
        ax.set_xlabel("Round")
        ax.set_ylabel("Principle ID")
        plt.yticks(rotation=0)
        self._save_figure(fig, "Belief_heatmap")
        plt.close(fig)

    def plot_ids_candidate_space(self, num_rounds_to_plot: int = 4):
        """
        Plot the Regret vs. Info Gain for candidate hypotheses at key rounds.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        self._configure_plot_style()

        data = []
        rounds_with_data = []
        for log in self.experiment_guidance_logs:
            if not log.information_ratios:
                continue

            rounds_with_data.append(log.round_number)
            for h, metrics in log.information_ratios.items():
                if "regret" in metrics and "info_gain" in metrics:
                    is_selected = h == log.selected_hypothesis
                    data.append(
                        {
                            "round": log.round_number,
                            "hypothesis": h,
                            "regret": metrics.get("regret"),
                            "info_gain": metrics.get("info_gain"),
                            "status": "Selected" if is_selected else "Candidate",
                        }
                    )

        if not data:
            logger.warning(
                "No information_ratios data found to plot IDS candidate space."
            )
            return

        df = pd.DataFrame(data).dropna()

        # Select representative rounds
        unique_rounds = sorted(list(set(rounds_with_data)))
        if len(unique_rounds) > num_rounds_to_plot:
            # Select first, last, and N-2 evenly spaced rounds
            indices = np.linspace(
                0, len(unique_rounds) - 1, num_rounds_to_plot, dtype=int
            )
            selected_rounds = [unique_rounds[i] for i in indices]
        else:
            selected_rounds = unique_rounds

        df_plot = df[df["round"].isin(selected_rounds)]
        if df_plot.empty:
            logger.warning("No valid data for selected rounds in IDS plot.")
            return

        # Use relplot for faceting by round
        g = sns.relplot(
            data=df_plot,
            x="info_gain",
            y="regret",
            hue="status",
            size="status",
            sizes={"Selected": 100, "Candidate": 20},
            style="status",
            markers={"Selected": "X", "Candidate": "o"},
            col="round",
            col_wrap=min(num_rounds_to_plot, 4),
            height=2.5,
            aspect=1.1,
            palette={"Selected": "red", "Candidate": "gray"},
            alpha=0.7,
        )

        g.set_axis_labels(
            "Information Gain ($I_t$)", "Instantaneous Regret ($\Delta_t$)"
        )
        g.fig.suptitle("IDS Candidate Space (Regret vs. Information)", y=1.03)
        self._save_figure(g.fig, "IDS_candidate_space")
        plt.close(g.fig)

    # ---------------------------------------------------------
    # OTHER THEORETICAL PLOTTING METHODS
    # ---------------------------------------------------------
    def plot_dual_uncertainty_phase_space(self):
        """
        Plots the trajectory of the system in the Dual-Uncertainty Phase Space.

        X-axis: U^EP (Epistemic Uncertainty over Principles - "Confusion")
        Y-axis: U^PH (Aleatoric/Generator Uncertainty - "Lack of Specificity")

        Theory:
        - High U_EP, High U_PH: Total Ignorance (Early game)
        - High U_EP, Low U_PH:  Conflicting Paradigms (System is debating which tight theory is right)
        - Low U_EP, High U_PH:  Broad Paradigm (System accepted a principle, but the principle is vague)
        - Low U_EP, Low U_PH:   Convergence (System knows the principle and the optimal hypothesis)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        if not self.theoretical_metrics_logs:
            logger.warning("No metrics to plot Phase Space.")
            return

        u_ep = np.array(
            [
                m.u_ep if m.u_ep is not None else np.nan
                for m in self.theoretical_metrics_logs
            ]
        )
        # Note: If u_ph is None (not logged), we try to derive it or skip
        u_ph = np.array(
            [
                m.u_ph if m.u_ph is not None else np.nan
                for m in self.theoretical_metrics_logs
            ]
        )
        rounds = np.array([m.round_number for m in self.theoretical_metrics_logs])

        # Filter valid data
        mask = np.isfinite(u_ep) & np.isfinite(u_ph)
        if np.sum(mask) < 2:
            logger.warning("Insufficient data (U_EP and U_PH) to plot Phase Space.")
            return

        x = u_ep[mask]
        y = u_ph[mask]
        r = rounds[mask]

        fig, ax = plt.subplots(figsize=(4.0, 3.5))

        # Plot the trajectory line
        sns.lineplot(x=x, y=y, sort=False, lw=1, color="gray", alpha=0.5, ax=ax)

        # Scatter points colored by time (Round)
        sc = ax.scatter(
            x, y, c=r, cmap="viridis", s=40, zorder=2, edgecolor="k", linewidth=0.5
        )

        # Add arrows to indicate direction of time
        # We plot arrows every few steps to avoid clutter
        if len(x) > 1:
            step = max(1, len(x) // 5)
            for i in range(0, len(x) - 1, step):
                ax.annotate(
                    "",
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6, lw=1),
                )

        # Annotate start and end
        ax.text(
            x[0], y[0], "Start", fontsize=8, fontweight="bold", ha="right", va="bottom"
        )
        ax.text(
            x[-1], y[-1], "Current", fontsize=8, fontweight="bold", ha="left", va="top"
        )

        # Quadrant Labels (Conceptual)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mid_x = (xlim[0] + xlim[1]) / 2
        mid_y = (ylim[0] + ylim[1]) / 2

        ax.text(
            xlim[0] * 0.95 + xlim[1] * 0.05,
            ylim[1] * 0.95,
            "Conflicting\nParadigms",
            fontsize=7,
            color="red",
            alpha=0.3,
            ha="left",
            va="top",
        )
        ax.text(
            xlim[1] * 0.95,
            ylim[1] * 0.95,
            "Total\nIgnorance",
            fontsize=7,
            color="red",
            alpha=0.3,
            ha="right",
            va="top",
        )
        ax.text(
            xlim[0] * 0.95 + xlim[1] * 0.05,
            ylim[0] * 0.95 + ylim[1] * 0.05,
            "Convergence",
            fontsize=7,
            color="green",
            alpha=0.3,
            ha="left",
            va="bottom",
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Round")

        ax.set_xlabel(r"$U^{EP}$ (Principle Uncertainty)")
        ax.set_ylabel(r"$U^{PH}$ (Hypothesis Uncertainty)")
        ax.set_title("Learning Dynamics Phase Space")

        self._save_figure(fig, "Dual_Uncertainty_Phase_Space")
        plt.close(fig)

    def plot_principle_survival_stream(self):
        """
        Plots a Streamgraph (Stackplot) of Principle Beliefs.

        This visualizes 'Paradigm Shifts'. Unlike the heatmap, this emphasizes
        the cumulative probability mass and competitive exclusion.
        It answers: "Did principle B replace principle A gradually or suddenly?"
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        self._configure_plot_style()

        if not self.principle_guidance_logs:
            return

        # 1. Aggregate beliefs per round
        records = []
        all_principles = set()

        # We need a dense matrix.
        round_map = {}  # round -> {pid: belief}

        for log in self.principle_guidance_logs:
            r = log.round_number
            round_map[r] = log.principle_beliefs.copy()
            all_principles.update(log.principle_beliefs.keys())

        if not round_map:
            return

        sorted_rounds = sorted(round_map.keys())
        sorted_principles = sorted(list(all_principles))

        # Build 2D array [n_principles, n_rounds]
        y_stack = np.zeros((len(sorted_principles), len(sorted_rounds)))

        for r_idx, r in enumerate(sorted_rounds):
            beliefs = round_map[r]
            total_b = sum(beliefs.values()) if beliefs else 1.0
            for p_idx, pid in enumerate(sorted_principles):
                val = beliefs.get(pid, 0.0)
                # Normalize just in case, for stackplot visualization
                y_stack[p_idx, r_idx] = val / total_b if total_b > 0 else 0

        fig, ax = plt.subplots(figsize=(5.0, 3.0))

        # Use a distinct colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_principles)))

        ax.stackplot(
            sorted_rounds, y_stack, labels=sorted_principles, colors=colors, alpha=0.85
        )

        ax.set_xlim(min(sorted_rounds), max(sorted_rounds))
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Round")
        ax.set_ylabel("Posterior Mass $p_t(P)$")
        ax.set_title("Principle Survival & Paradigm Shifts")

        # Dynamic Legend: Only show principles that ever exceeded 10% mass
        # to avoid cluttering the legend with dead hypotheses.
        handles, labels = ax.get_legend_handles_labels()
        significant_indices = [i for i, row in enumerate(y_stack) if np.max(row) > 0.1]

        if significant_indices:
            sig_handles = [handles[i] for i in significant_indices]
            sig_labels = [labels[i] for i in significant_indices]
            ax.legend(
                sig_handles,
                sig_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=7,
                title="Dominant Principles",
            )

        self._save_figure(fig, "Principle_Survival_Stream")
        plt.close(fig)

    def plot_learning_efficiency_frontier(self):
        """
        Plots Cumulative Information Gain vs. Cumulative Regret.

        UPDATED: Normalizes values to [0, 1] relative to the maximum observed
        values to visualize the 'shape' of efficiency without unit mismatch.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        data = self._get_empirical_metrics()
        if data is None:
            logger.warning("No valid experiment data to plot Efficiency Frontier.")
            return

        cum_regret = data["cum_regret"]
        cum_info = data["cum_info"]
        rounds = data["rounds"]

        # --- NORMALIZATION STEP ---
        # Normalize to [0, 1] based on the final/max accumulated value
        # This prevents the 1e-8 scale from making the plot look weird,
        # though you should still investigate why InfoGain is so low.
        max_regret = cum_regret.max() if cum_regret.max() > 0 else 1.0
        max_info = cum_info.max() if cum_info.max() > 0 else 1.0

        norm_regret = cum_regret / max_regret
        norm_info = cum_info / max_info

        fig, ax = plt.subplots(figsize=(3.5, 3.0))

        # Plot the normalized frontier curve
        sns.lineplot(x=norm_regret, y=norm_info, sort=True, lw=1.5, color="teal", ax=ax)

        # Scatter points colored by round
        sc = ax.scatter(norm_regret, norm_info, c=rounds, cmap="magma", s=30, zorder=2)

        # Add diagonal reference line (The "Unit Efficiency" Baseline)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8, label="Linear Baseline")

        # Labels
        ax.set_xlabel(r"Normalized Cum. Regret (Cost) \%")
        ax.set_ylabel(r"Normalized Cum. Info Gain (Value) \%")

        # Add text indicating the raw scales (so we don't lose that context)
        # Using [cite: 26] 1e-8 from your upload as context for why we do this
        ax.text(
            0.05,
            0.95,
            f"Max Regret: {max_regret:.1f}",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.text(
            0.05, 0.90, f"Max Info: {max_info:.2e}", transform=ax.transAxes, fontsize=6
        )

        ax.set_title(f"Learning Efficiency (Normalized)")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Round")

        self._save_figure(fig, "Learning_Efficiency_Frontier_Norm")
        plt.close(fig)

    def plot_ids_dynamics(self):
        """
        Plots the evolution of the Information Ratio (IDS Trade-off).

        CORRECTION: Recalculates ratio using Empirical Regret.
        Y-axis: rho_t = Delta_t^2 / I_t
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self._configure_plot_style()

        data = self._get_empirical_metrics()
        if data is None:
            logger.warning("No valid experiment data to plot IDS Dynamics.")
            return

        rounds = data["rounds"]
        ratios = data["ids_ratios"]

        # Filter out massive outliers for cleaner plotting if I_t is tiny
        # We clip at a reasonable upper bound for visualization (e.g. 10^4)
        ratios_clipped = np.clip(ratios, 1e-6, 1e5)

        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        sns.lineplot(x=rounds, y=ratios_clipped, color="#d62728", lw=1.2, ax=ax)

        # Use Log scale because ratios span orders of magnitude
        ax.set_yscale("log")

        ax.set_xlabel("Round")
        ax.set_ylabel(r"Info Ratio $\rho_t = \Delta_t^2 / I_t$")
        ax.set_title("Exploration-Exploitation Dynamics")

        # Threshold line (Heuristic: 1.0 is a common theoretical boundary)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

        # Annotations (Check if limits allow them to be visible)
        y_min, y_max = ax.get_ylim()
        if y_max > 10:
            ax.text(
                rounds[0],
                2.0,
                "Exploitation Dominant",
                fontsize=6,
                color="gray",
                va="bottom",
            )
        if y_min < 0.1:
            ax.text(
                rounds[0],
                0.5,
                "Exploration Dominant",
                fontsize=6,
                color="gray",
                va="top",
            )

        self._save_figure(fig, "IDS_Dynamics_Rho")
        plt.close(fig)

    # -----------------------------
    # Utility methods
    # -----------------------------
    def clear_computed_metrics(self) -> None:
        """Clear previously computed theoretical metrics (keeps the raw logs)."""
        self.theoretical_metrics_logs = []
        self._cumulative_regret = 0.0
        self._cumulative_information = 0.0

    def export_logs(self, filename_prefix: str = "pievo_metrics") -> None:
        """
        Simple on-disk dump of logs as python reprs.
        You can replace with JSON/pickle or a more structured format as needed.
        """
        import json

        out = {
            "principle_logs": [asdict(x) for x in self.principle_guidance_logs],
            "hypothesis_logs": [asdict(x) for x in self.hypothesis_guidance_logs],
            "experiment_logs": [asdict(x) for x in self.experiment_guidance_logs],
            "theoretical_metrics": [asdict(x) for x in self.theoretical_metrics_logs],
        }
        path = os.path.join(self.log_dir, f"{filename_prefix}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        logger.debug("Exported logs to %s", path)


if __name__ == "__main__":
    track = FlowTracker(
        target_dir=f"/home/mellen/Desktop/PiEvo/reported_A_PiEvo_MBO_EXPORT_sigma_0d2",
        load_from_file=f"/home/mellen/Desktop/PiEvo/reported_A_PiEvo_MBO_EXPORT_sigma_0d2/metrics/pievo_metrics.json",
    )
    track.compute_all_metrics()
