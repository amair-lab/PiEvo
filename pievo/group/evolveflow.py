import os
import json
import logging
import math
import warnings

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Sequence, Tuple
from datetime import datetime

from pievo.word.bayesian import GaussianProcessModel
from pievo.word.embedding import EmbeddingService
from pievo.word.extraction import extract_json_blocks
from pievo.word.instruction import get_hypothesis_guidance_prompt, get_experiment_guidance_prompt, get_principle_guidance_prompt
from pievo.word.utils import sanitize, get_submission_hash

os.environ["KMP_DUPLICATE_LIB_OK"] = "true"

from autogen_agentchat.messages import ChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Public Constants
EXPERIMENT_SAVED_FNAME = "experiment.json"
SUBMISSION_AND_GUIDANCE_FNAME = "submission_and_guidance.json"
SUBMISSION_NONE_GUIDANCE_FNAME = "submission_none_guidance.json"

class PiEvo:
    def __init__(
            self,
            task: str,
            client: OpenAIChatCompletionClient,
            output_dir: str,
            output_file: str = SUBMISSION_AND_GUIDANCE_FNAME,
            sigma: float = 0.1,
            anomaly_threshold: float = 0.85,
            warm_up_rounds: int = 8,
            new_principle_prior_mass: float = 1e-3,
            off_pievo: bool = False,
    ):
        self.submissions: List[Dict[str, Any]] = []
        self.output_dir = output_dir
        self.output_file = output_file
        self.util_client = client
        self.task = task
        self.is_continuous_search: bool = "parameter" in task.lower()
        logger.warning("Currently, PiEvo automatically detects the task type by searching `parameter` keyword in self.task. This may cause unexpected errors. ")

        # --- Algorithm State ---
        self.principles: Dict[str, str] = {}
        self.principle_beliefs: Dict[str, float] = {}
        self.principle_priors: Dict[str, float] = {}
        self.history: List[Tuple[str, float]] = []
        self.reward_history: List[float] = []
        self.principles_rationals: Dict[str, dict] = {}
        self.is_exploitation_phase: bool = False
        self.exploitation_candidates: List[dict] = []

        # --- Parameters ---
        self.sigma = sigma
        self.anomaly_threshold = anomaly_threshold
        self.warm_up_rounds = warm_up_rounds
        self.new_principle_prior_mass = new_principle_prior_mass

        self.embedding_service = EmbeddingService()
        self._y_mean = 0.0
        self._y_std = 1.0

        # --- GP Models ---
        self.gp_models: Dict[str, "GaussianProcessModel"] = {}
        self.feature_cache: Dict[Tuple[str, str], np.ndarray] = {}

        # --- Caches ---
        self.prediction_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self.optimal_prediction_cache: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
        self.optimal_value_star: Optional[float] = None

        # --- State ---
        self._processed_submission_hashes = set()
        self.round_by_PHE_order_before_P: int = 0
        self.status: str = "Initializing"

        self.hypothesis_form = {}
        self.off_pievo = off_pievo

    def _process_message(self, message_content: str, source_agent: str) -> List[Dict[str, Any]]:
        extracted_submissions = extract_json_blocks(message_content, source_agent)
        new_submissions = []
        for submission in extracted_submissions:
            submission_hash = get_submission_hash(submission)

            if source_agent == "principle":
                principle_id = list(submission["json_data"].items())[0][0]
                principle_rationals = list(submission["json_data"].items())[0][1]["RATIONAL"]
                self.principles_rationals[principle_id] = principle_rationals

            if submission_hash not in self._processed_submission_hashes:
                self.submissions.append(submission)
                self._processed_submission_hashes.add(submission_hash)
                new_submissions.append(submission)
        if new_submissions:
            self.save_to_file()
        return new_submissions

    def save_to_file(self):
        with open(self.output_file, "w") as f:
            json.dump({
                "metadata": {"last_updated": datetime.now().isoformat()},
                "submissions": self.submissions,
            }, f, indent=4)

    def clear_caches(self):
        self.prediction_cache.clear()
        self.optimal_prediction_cache.clear()
        self.feature_cache.clear()

    def _update_scaler(self):
        if len(self.history) < 2:
            self._y_mean = 0.0
            self._y_std = 1.0
            return
        outcomes = [y for h, y in self.history if y != 0.0]  # Filter out failed experiments
        if not outcomes:
            self._y_mean = 0.0
            self._y_std = 1.0
            return
        self._y_mean = np.mean(outcomes)
        std = np.std(outcomes)
        self._y_std = std if std > 1e-9 else 1.0

        # We do not update sigma here is to:
        # Ensure that self.sigma = 0.01 means "The observation noise is 1% of the historical standard deviation."

    def _normalize_y(self, y: float) -> float:
        """Raw -> Standardized"""
        return (y - self._y_mean) / self._y_std

    def _denormalize_y(self, y_norm: float) -> float:
        """Standardized -> Raw"""
        return (y_norm * self._y_std) + self._y_mean

    async def _calculate_bayesian_regret(self, hypothesis: str) -> float:
        """Academic formula: Regret(h) = E[v*(P)] - E[mu_P(h)]. Used for IDS."""
        if not self.principle_beliefs: return 0.0
        
        # Weighted optimal value across principles
        v_star = sum(self.principle_beliefs[pid] * await self._estimate_principle_optimal_value(pid)
                     for pid in self.principle_beliefs)
        
        # Weighted predicted value for this candidate
        v_candidate = sum(self.principle_beliefs[pid] * await self._estimate_candidate_value_from_gp_model(hypothesis, pid)
                          for pid in self.principle_beliefs)
        
        return max(0.0, v_star - v_candidate)

    def gather_submission_from_message(self, messages: Sequence[ChatMessage]) -> Optional[str]:
        for message in messages:
            if (
                    hasattr(message, "content")
                    and hasattr(message, "source")
                    and message.content
                    and message.source
            ):
                content_str = str(message.content)
                if "```json" in content_str.lower():
                    extracted_submissions = self._process_message(
                        message_content=content_str,
                        source_agent=message.source,
                    )
                    if extracted_submissions:
                        if self.is_exploitation_phase and message.source == "experiment":
                            logger.error(f"DEBUG: CANDIDATE ADDED FOR EXPLOITATION:  {extracted_submissions}")
                            self.exploitation_candidates += extracted_submissions
                        return message.source
        return None

    def _extract_features(self, hypothesis: str, principle_id: str) -> np.ndarray:
        """
        1. Basic and easy-to-get features:
            a. Dot Product of h_emb and p_emb
            b. Euclidean Dist of h_emb and p_emb
        2. LLM-as-Judge Features:
            a. Alignment Score: Does the parameter setting of the Hypothesis strictly follow the qualitative description of the Principle?
            b. Derivation Logic: Is the logical chain in the rational part of the Hypothesis complete and consistent?
        """
        cache_key = (hypothesis, principle_id)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        embed_func = self.embedding_service.embed_with_local_model

        # Get embeddings of Hypothesis and Principle
        h_emb = np.array(embed_func(hypothesis))
        principle_text = self.principles.get(principle_id)
        p_emb = np.array(embed_func(principle_text))
        h_emb = h_emb / (np.linalg.norm(h_emb) + 1e-9)
        p_emb = p_emb / (np.linalg.norm(p_emb) + 1e-9)

        # Base features of similarity.
        phi_1_projection = np.dot(h_emb, p_emb)
        phi_2_euclidean_dist = np.linalg.norm(h_emb - p_emb)
        _base_features = np.array([phi_1_projection, phi_2_euclidean_dist])


        features = np.concatenate([_base_features, ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        self.feature_cache[cache_key] = features
        return features

    def _extract_principles(self) -> List[str]:
        """Extracts principles from submissions and identifies newly discovered ones."""
        new_principle_ids = []
        for submission in self.submissions:
            if submission["source_agent"] == "principle":
                json_data = submission["json_data"]
                if isinstance(json_data, dict):
                    for key, value in json_data.items():
                        if key.startswith("principle_"):
                            principle_text = (
                                value.get("PRINCIPLE_SUBMISSION", "")
                                if isinstance(value, dict)
                                else str(value)
                            )
                            if key not in self.principles:
                                self.principles[key] = principle_text
                                new_principle_ids.append(key)
        return new_principle_ids

    def _handle_new_principles(self, new_principle_ids: List[str]):
        """
        Implements Coherent Augmentation by assigning a prior p_0(P_new)
        to newly discovered principles.
        """
        if not new_principle_ids:
            return

        if not self.principle_priors:  # First principles
            prob = self.new_principle_prior_mass
            for pid in self.principles:
                self.principle_priors[pid] = prob
        else:
            for new_pid in new_principle_ids:
                if new_pid not in self.principle_priors:
                    # Assign the initial belief score.
                    self.principle_priors[new_pid] = self.new_principle_prior_mass

    def _extract_history(self) -> None:
        """Extracts verified (hypothesis, outcome) pairs from submissions to form H_{t-1}."""
        # This implementation rebuilds history each time for simplicity.
        # For performance, it could be made incremental.
        hypotheses = {}
        outcomes = {}
        for submission in self.submissions:
            if submission["source_agent"] == "hypothesis":
                for key, value in submission["json_data"].items():
                    if key.startswith("HYPOTHESIS_") and "candidate" in value:
                        hypotheses[value["candidate"]] = value
        for submission in self.submissions:
            if submission["source_agent"] == "experiment":
                if isinstance(submission["json_data"], list):
                    for result in submission["json_data"]:
                        if "candidate" in result and isinstance(
                                result.get("outcome"), (int, float)
                        ):
                            outcomes[result["candidate"]] = result["outcome"]
        self.history = []
        for hypothesis, outcome in outcomes.items():
            if hypothesis in hypotheses:
                self.history.append((hypothesis, outcome))

    async def _get_prediction(self, hypothesis: str, principle_id: str, return_normalized: bool = False) -> Optional[float]:
        """
        Gets f_P(h), the predicted outcome for a hypothesis h under principle P.
        Uses Bayesian linear model trained on historical data only.
        """
        cache_key = (hypothesis, principle_id)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        if principle_id not in self.gp_models:
            logger.error(
                f"Prediction failed: GP model for {principle_id} does not exist. "
            )
            self.prediction_cache[cache_key] = None
            return None

        try:
            model = self.gp_models[principle_id]
            features = self._extract_features(hypothesis, principle_id)

            if len(features) != model.feature_dim:
                logger.error(
                    f"Feature dimension mismatch for {principle_id}: "
                    f"Model expected {model.feature_dim}, got {len(features)}. "
                    f"This may indicate a state error."
                )
                self.prediction_cache[cache_key] = None

            # Use trained Bayesian model
            mean_norm, variance = model.predict(features)

            if return_normalized:
                return mean_norm

            prediction = self._denormalize_y(mean_norm)

        except Exception as e:
            logger.warning(f"Prediction failed for ({hypothesis}, {principle_id}): {e}")
            prediction = None

        self.prediction_cache[cache_key] = prediction
        return prediction

    async def _likelihood(self, outcome: float, hypothesis: str, principle_id: str, is_normalized_input: bool = False) -> float:
        """
        Computes the likelihood p(y|h,P) using a Gaussian model.
        Uses Bayesian model's predictive distribution when available.
        Now properly accounts for both model and observational uncertainty.
        """
        effective_noise_var = self.sigma ** 2

        if is_normalized_input:
            y_norm = outcome
        else:
            y_norm = self._normalize_y(outcome)

        if (
                principle_id in self.gp_models
                and self.gp_models[principle_id].n_observations > 0
        ):
            # Use GP model's predictive distribution
            model = self.gp_models[principle_id]

            features = self._extract_features(hypothesis, principle_id)
            mean, model_variance = model.predict(features)

            # Total variance = model uncertainty + observational noise
            total_variance = model_variance + effective_noise_var
            if total_variance <= 0:
                total_variance = effective_noise_var  # Fallback to default noise

            # Compute likelihood with proper total uncertainty
            exponent = -0.5 * ((y_norm - mean) ** 2) / total_variance
            factor = 1.0 / math.sqrt(2 * math.pi * total_variance)
            likelihood = factor * math.exp(exponent)
            return max(likelihood, 1e-12)  # Use more conservative epsilon

        # Fallback to point prediction + fixed noise
        prediction = await self._get_prediction(hypothesis, principle_id, return_normalized=True)
        if prediction is None:
            return 0.0

        variance = effective_noise_var
        exponent = -0.5 * ((y_norm - prediction) ** 2) / variance
        factor = 1.0 / math.sqrt(2 * math.pi * variance)
        likelihood = factor * math.exp(exponent)
        return max(likelihood, 1e-12)  # Use more conservative epsilon

    async def _update_beliefs_from_full_history(self) -> None:
        """
        Correctly computes the posterior p_t(P) ∝ p_0(P) Π p(y_s|h_s, P)
        over the *full history* for *all* principles in the working set.
        Handles the case of no history by setting beliefs to priors.
        """
        if not self.principles:
            self.principle_beliefs = {}
            return

        # If there is no history, beliefs are the priors.
        if not self.history:
            if not self.principle_priors:
                # Initialize uniform priors if they don't exist
                prob = 1e-2
                self.principle_priors = {pid: prob for pid in self.principles}

            # Set beliefs from priors and normalize
            self.principle_beliefs = self.principle_priors.copy()
            total_prior = sum(self.principle_beliefs.values())
            if total_prior > 0:
                for pid in self.principle_beliefs:
                    self.principle_beliefs[pid] /= total_prior
            else:  # Fallback to uniform if sum is zero
                prob = 1e-2
                self.principle_beliefs = {pid: prob for pid in self.principles}
            return

        # --- Emphasized Logic Change ---
        # 1. Start from the fixed prior p_0(P)
        log_posteriors = {
            pid: math.log(max(1e-12, self.principle_priors.get(pid, 1e-6)))
            for pid in self.principles.keys()
        }

        # 2. Apply likelihood from ALL history items
        for h, y in self.history:
            if y == 0.0:  # Skip failed experiments
                continue
            for pid in self.principles.keys():
                try:
                    likelihood = await self._likelihood(y, h, pid)
                    log_posteriors[pid] += math.log(max(likelihood, 1e-12))
                except Exception:
                    log_posteriors[pid] -= 50  # Penalize failure

        # 3. Normalize using log-sum-exp for stability
        log_sum_exp = float('-inf')
        for log_prob in log_posteriors.values():
            log_sum_exp = np.logaddexp(log_sum_exp, log_prob)

        if np.isinf(log_sum_exp):
            # All posteriors are zero, fallback to uniform
            logger.warning("All posteriors are zero, resetting to uniform.")
            prob = 1e-2
            self.principle_beliefs = {pid: prob for pid in self.principles}
            return

        # 4. Compute final posterior beliefs
        self.principle_beliefs = {
            pid: math.exp(log_post - log_sum_exp)
            for pid, log_post in log_posteriors.items()
        }

        await self._train_gp_models()  # Keep GP training incremental
        self._last_belief_update_history_index = len(self.history)

    async def _train_gp_models(self) -> None:
        """
        Incrementally updates GP models with new historical data.
        Ensures new principles are "back-filled" with all historical data.
        Ensures models are created even if history is empty (t=0 case).
        """
        actual_feature_dim = 2

        if not self.principles:
            return

        if self.history:
            sample_hypothesis, _ = self.history[0]
            sample_principle_id = next(iter(self.principles.keys()), None)
            if sample_principle_id:
                try:
                    sample_features = self._extract_features(sample_hypothesis, sample_principle_id)
                    actual_feature_dim = len(sample_features)
                    logger.debug(f"GP Training: Determined feature dimension is {actual_feature_dim}")
                except Exception as e:
                    logger.warning(f"Could not determine actual feature dim, falling back to default: {e}")

        # 1. Clear existing models to ensure we retrain with current Scaler (_y_mean/_y_std)
        self.gp_models.clear()

        # 2. Re-train all models with ALL history using CURRENT scaler, this will be safe.
        for principle_id in self.principles.keys():
            self.gp_models[principle_id] = GaussianProcessModel(
                principle_id=principle_id,
                log_dir=self.output_dir,
                feature_dim=actual_feature_dim,
            )

            # Batch update (more efficient than loop if supported, otherwise loop)
            # Assuming incremental update is fast enough:
            for h, y in self.history:
                if y != 0.0:
                    try:
                        features = self._extract_features(h, principle_id)
                        y_norm = self._normalize_y(y)  # Uses CURRENT scaler
                        self.gp_models[principle_id].update(features, y_norm)
                    except Exception as e:
                        logger.warning(f"GP Update failed: {e}")

        self._last_processed_history_index = len(self.history)

    async def update_pievo_state(self) -> None:
        logger.debug("🧠 Updating PiEvo state...")
        old_history_len = len(self.history)

        # Extract the new principle from the self.submissions, and assign the initial belief for handling.
        new_principle_ids = self._extract_principles()
        self._handle_new_principles(new_principle_ids)
        self._extract_history()

        self._update_scaler()

        # Invalidate caches if state changed
        if new_principle_ids or len(self.history) != old_history_len:
            self.clear_caches()

        # As long as we have principles, we train it by _train_gp_models
        # If history is blank (t=0), it also needs the blank GP model.
        if self.principles:
            await self._train_gp_models()
            await self._update_beliefs_from_full_history()
            if self.history:
                await self._update_regret_and_rewards()

    def _is_warm_up_phase(self) -> bool:
        """Check if we're still in the warm-up phase"""
        return len(self.history) < self.warm_up_rounds

    async def _select_hypothesis_warm_up(self, candidate_hypotheses: List[str]) -> str:
        """
        Select hypothesis during warm-up phase using uncertainty sampling or random selection.
        This helps accumulate diverse initial data for GP models.
        """
        # Uncertainty sampling: select hypothesis with the highest prediction variance
        best_hypothesis = None
        best_uncertainty = -1.0

        for h in candidate_hypotheses:
            total_uncertainty = 0.0
            valid_principles = 0

            for pid in self.principles.keys():
                try:
                    if pid in self.gp_models and self.gp_models[pid].n_observations > 0:
                        # Use GP model uncertainty
                        features = self._extract_features(h, pid)
                        _, variance = self.gp_models[pid].predict(features)
                        uncertainty = np.sqrt(variance)
                    else:
                        # Use default high uncertainty for untrained models
                        uncertainty = 1.0

                    total_uncertainty += uncertainty
                    valid_principles += 1

                except Exception as e:
                    logger.debug(f"Error computing uncertainty for {pid}: {e}")
                    continue

            # Average uncertainty across principles
            avg_uncertainty = total_uncertainty / max(1, valid_principles)

            if avg_uncertainty > best_uncertainty:
                best_uncertainty = avg_uncertainty
                best_hypothesis = h

            if best_hypothesis:
                logger.debug(f"Warm-up (uncertainty): Selected '{best_hypothesis[:30]}...' (uncertainty: {best_uncertainty:.3f})")
                return best_hypothesis
            else:
                # Fallback to random if uncertainty computation fails
                import random
                selected = random.choice(candidate_hypotheses)
                logger.debug(f"Warm-up (fallback): Selected '{selected[:30]}...'")
                return selected

        else:
            # Fallback to first candidate
            return candidate_hypotheses[0]

    def _calculate_optimal_value_star(self) -> Optional[float]:
        """
        Calculate v* = max_h r(h, P*) - the optimal value under the true principle.
        Based on Eq. in theoretical_framework.tex

        Returns:
            Optimal value as the maximum observed reward from experiment outcomes
        """
        if self.reward_history:
            try:
                valid_rewards = [r for r in self.reward_history if r is not None]
                if valid_rewards:
                    max_reward = max(valid_rewards)
                    logger.debug(f"Calculated optimal_value_star from reward_history: {max_reward}")
                    return float(max_reward)
            except (ValueError, TypeError, StopIteration):
                logger.error("Failed to calculate `optimal_value_star` from `reward_history`.")

        logger.debug(f"Could not calculate optimal_value_star. reward_history length: {len(self.reward_history) if self.reward_history else 0}")
        return None

    def _calculate_optimal_hypothesis_for_principle(self, principle_id: str) -> Optional[Tuple[str, float]]:
        """
        Calculate h*(P) = argmax_h r(h, P) - the optimal hypothesis for a given principle.
        Theory: For each principle P, find the hypothesis that maximizes reward under that principle.

        Args:
            principle_id: ID of the principle P

        Returns:
            Tuple of (optimal_hypothesis, optimal_reward) or None if not available
        """
        if principle_id not in self.gp_models:
            return None

        model = self.gp_models[principle_id]
        if model.n_observations == 0:
            return None

        # Find the hypothesis in history that performed best under this principle
        best_hypothesis = None
        best_predicted_reward = -float('inf')

        # Check cache first
        if principle_id in self.optimal_prediction_cache:
            cached_hypothesis, cached_reward = self.optimal_prediction_cache[principle_id]
            if cached_hypothesis is not None and cached_reward is not None:
                return cached_hypothesis, cached_reward

    async def _update_regret_and_rewards(self) -> None:
        """Minimalist reward history update."""
        self.reward_history = [y for h, y in self.history]

    async def _estimate_principle_optimal_value(self, principle_id: str) -> float:
        """Simple MLE-based estimate of max_h f_P(h) based only on history."""
        if principle_id not in self.gp_models: return self._y_mean
        model = self.gp_models[principle_id]
        if model.n_observations == 0: return self._y_mean
        
        preds = []
        for h, _ in self.history:
            m, _ = model.predict(self._extract_features(h, principle_id))
            preds.append(m)
        return self._denormalize_y(max(preds)) if preds else self._y_mean

    async def _estimate_candidate_value_from_gp_model(self, hypothesis: str, principle_id: str) -> float:
        """Simple GP point prediction."""
        if principle_id not in self.gp_models: return self._y_mean
        m, _ = self.gp_models[principle_id].predict(self._extract_features(hypothesis, principle_id))
        return self._denormalize_y(m)

    async def _estimate_information_gain_bald(self, hypothesis: str) -> float:
        """Simplified BALD: Reduction in principle entropy."""
        current_entropy = self._calculate_principle_uncertainty()
        # Monte Carlo sampling of future entropy
        future_entropies = []
        for _ in range(5): # Scaled down
             sampled_y = await self._sample_outcome_marginalized(hypothesis)
             future_entropies.append(await self._calculate_hypothetical_posterior_entropy(hypothesis, sampled_y))
        return max(0.0, current_entropy - np.mean(future_entropies))

    async def _sample_outcome_marginalized(self, hypothesis: str) -> float:
        """Sample y ~ p(y|h) = sum_P p(P) p(y|h,P)"""
        if not self.principle_beliefs: return 0.0
        pids = list(self.principle_beliefs.keys())
        probs = list(self.principle_beliefs.values())
        sampled_pid = np.random.choice(pids, p=probs)
        pred_m, pred_v = self.gp_models[sampled_pid].predict(self._extract_features(hypothesis, sampled_pid))
        return pred_m + np.random.normal(0, np.sqrt(pred_v + self.sigma**2))

    async def _calculate_hypothetical_posterior_entropy(self, hypothesis: str, sampled_y: float) -> float:
        """Estimate entropy of posterior belief if sampled_y was observed."""
        log_posts = {pid: math.log(max(1e-12, self.principle_beliefs[pid])) for pid in self.principles}
        for pid in self.principles:
            likelihood = await self._likelihood(sampled_y, hypothesis, pid, is_normalized_input=True)
            log_posts[pid] += math.log(max(likelihood, 1e-12))
        
        max_log = max(log_posts.values())
        exp_posts = np.array([math.exp(v - max_log) for v in log_posts.values()])
        new_beliefs = exp_posts / exp_posts.sum()
        return -np.sum(new_beliefs * np.log(new_beliefs + 1e-12))

    def _calculate_principle_uncertainty(self) -> float:
        """Shannon Entropy H(p_t(P))."""
        beliefs = np.array(list(self.principle_beliefs.values()))
        beliefs = beliefs[beliefs > 1e-12]
        return -np.sum(beliefs * np.log(beliefs)) if len(beliefs) > 0 else 0.0

    async def _detect_anomalies(self) -> List[Tuple[str, float, float, float]]:
        """Identify observations y that are unlikely under MAP principle."""
        if not self.history or not self.principle_beliefs: return []
        map_pid = max(self.principle_beliefs, key=self.principle_beliefs.get)
        if map_pid not in self.gp_models: return []
        
        anomalies = []
        for h, y in self.history:
            y_norm = self._normalize_y(y)
            pred_m, pred_v = self.gp_models[map_pid].predict(self._extract_features(h, map_pid))
            surprisal = 1 - math.exp(-math.sqrt(((y_norm - pred_m)**2) / (pred_v + self.sigma**2)))
            if surprisal > self.anomaly_threshold:
                anomalies.append((h, y, self._denormalize_y(pred_m), surprisal))
        return anomalies
    def _calculate_adaptive_anomaly_threshold(self) -> float:
        """Return the fixed threshold for minimalist implementation."""
        return self.anomaly_threshold

    async def get_principle_guidance(self) -> str:
        """
        Provides coordination guidance to the PrincipleAgent based on anomaly detection.
        This method also marks the start of a new P-H-E cycle and handles logging for the previous cycle.
        """
        # This method marks the beginning of a new round.
        # First, update state with results from the previous round's experiment.
        self.status = "Analyzing Anomalies & Generating Principles"
        await self.update_pievo_state()

        # Start of the new round
        self.round_by_PHE_order_before_P += 1
        anomalies = await self._detect_anomalies()

        try:
            top_principle = self.principles.get(
                max(self.principle_beliefs, key=self.principle_beliefs.get)
            )
        except ValueError:
            top_principle = ""

        adaptive_threshold = self._calculate_adaptive_anomaly_threshold()

        guidance_type, guidance_message = get_principle_guidance_prompt(
            anomalies=anomalies,
            num_principles=len(self.principles),
            principle_beliefs=self.principle_beliefs,
            top_principle_text=top_principle,
            principles=self.principles,
        )

        self.submissions.append({
            "source_agent": "principle_guidance",
            "guidance": guidance_message
        })
        return guidance_message

    async def get_hypothesis_guidance(self) -> str:
        """
        Provides coordination guidance to the HypothesisAgent.
        """
        self.status = "Generating Hypotheses"
        await self.update_pievo_state()

        selection_method = "map_based_guidance"

        top_principle_id = None
        top_principle_text = ""
        second_best_principle_text = None

        if self.principle_beliefs:
            sorted_beliefs = sorted(self.principle_beliefs.items(), key=lambda item: item[1], reverse=True)
            if sorted_beliefs:
                top_principle_id = sorted_beliefs[0][0]
                top_principle_text = self.principles.get(top_principle_id, "")
                if len(sorted_beliefs) > 1:
                    second_best_principle_id = sorted_beliefs[1][0]
                    second_best_principle_text = self.principles.get(second_best_principle_id)

        candidates_done = []
        for submission in self.submissions:
            if submission["source_agent"] == "experiment":
                if submission["json_data"] and isinstance(submission["json_data"], list):
                    candidate = submission["json_data"][0].get("candidate")
                    outcome = submission["json_data"][0].get("outcome")
                    if candidate:
                        candidates_done.append({"candidate": candidate, "outcome": outcome})

        guidance_type, guidance_message = get_hypothesis_guidance_prompt(
            task=self.task,
            top_principle_text=top_principle_text,
            second_best_principle_text=second_best_principle_text,
            principle_beliefs=self.principle_beliefs,
            num_principles=len(self.principles),
            exploitation_candidates=self.exploitation_candidates,
            principles=self.principles,
            candidates_done=candidates_done
        )

        if guidance_type == "EXPLOITATION":
            self.is_exploitation_phase = True
        else:
            self.is_exploitation_phase = False

        self.submissions.append({
            "source_agent": "hypothesis_guidance",
            "guidance": guidance_message
        })
        return guidance_message

    async def get_experiment_guidance(self, candidate_hypotheses: List[str]) -> str:
        """
        Provides coordination guidance to the ExperimentAgent using Information-Directed Sampling or Outcome Optimization.
        """
        self.status = "Selecting Experiment Candidate"
        await self.update_pievo_state()

        logger.warning(f"GUIDANCE of candidates scope: {candidate_hypotheses}")

        # === Robustness Check: Handle Empty Candidate List ===
        if not candidate_hypotheses:
            logger.error("get_experiment_guidance was called with an empty list of candidates. No hypothesis can be selected.")

            guidance_message = ("No untested candidate hypotheses provided. Cannot select an experiment. "
                                "Instruct the Hypothesis Agent to provide **diverse candidates** for testing. ")
            return guidance_message

        information_ratios = {}

        # === Prepare all the values needed ===
        # 1. the predicted outcome for a hypothesis h under principle P.
        predictions_norm: Dict[Tuple[str, str], float] = {}
        for h in candidate_hypotheses:
            for pid in self.principles:
                pred = await self._get_prediction(h, pid, return_normalized=True)
                predictions_norm[(h, pid)] = pred if pred is not None else 0.0

        # 2. Compute E[r(h)] = sum( p(P) * f_P(h) )
        candidate_rewards_norm: Dict[str, float] = {}  # h -> E[r(h)]
        for h in candidate_hypotheses:
            expected_r = 0.0
            for pid, belief in self.principle_beliefs.items():
                expected_r += belief * predictions_norm[(h, pid)]
            candidate_rewards_norm[h] = expected_r

        # WARM-UP PHASE:
        if self._is_warm_up_phase():
            selection_method = "warm_up_uncertainty"
            selected_hypothesis = await self._select_hypothesis_warm_up(candidate_hypotheses)
            guidance_message = get_experiment_guidance_prompt(selected_hypothesis, self.task)

        # ==============  EXPLOITATION Phase  ==============
        elif self.is_exploitation_phase:
            selection_method = "exploit_minimize_regret"
            best_hypothesis = None
            best_regret = float("inf")

            for h in candidate_hypotheses:
                # Note: Here the id_regret is now Raw Scale (e.g., 100.0)
                id_regret = await self._calculate_bayesian_regret(h)
                information_ratios[h] = {"regret": float(id_regret), "info_gain": None, "ratio": None}

                if id_regret < best_regret:
                    best_regret = id_regret
                    best_hypothesis = h

            selected_hypothesis = best_hypothesis
            guidance_message = get_experiment_guidance_prompt(selected_hypothesis, self.task)

        # ============== EXPLORATION Phase (IDS) ==============
        else:
            # Objective: min (E[v*] - E[r(h)])^2 / I(h)
            selection_method = "explore_information_directed"

            # The key is to compute E[v*]'s proxy
            # E[v*] approx sum( p(P) * max_h(f_P(h)) )
            expected_optimal_value_norm: float = 0.0
            for pid, belief in self.principle_beliefs.items():
                # Find the h that P believes the best (in C_t)
                best_reward_for_this_p = -float('inf')
                for h in candidate_hypotheses:
                    best_reward_for_this_p = max(best_reward_for_this_p, predictions_norm[(h, pid)])

                expected_optimal_value_norm += belief * best_reward_for_this_p

            # Compute the IDS regret
            best_hypothesis_ids = None
            best_ratio = float("inf")
            regrets = {}

            for h in candidate_hypotheses:
                # Use E[v*] to compute the regret
                id_regret_norm = expected_optimal_value_norm - candidate_rewards_norm[h]
                id_regret_norm = max(0.0, id_regret_norm)
                regrets[h] = id_regret_norm

                info_gain = await self._estimate_information_gain_bald(h)

                if info_gain > 1e-6:
                    ratio = (id_regret_norm ** 2) / info_gain
                    ratio = min(ratio, 1e3)  # keep a large value bar
                    # We used Norm for the Math, but we log the RAW value for consistency
                    # Scale Norm Regret back to Raw Scale for the log
                    # Since $\text{Regret} = y^* - y$, and $y = \sigma y_{norm} + \mu$, then $\Delta y = \sigma \Delta y_{norm}$. Only times self._y_std is correct.
                    id_regret_raw = id_regret_norm * self._y_std

                    information_ratios[h] = {
                        "regret": float(id_regret_raw),  # LOG RAW VALUE
                        "regret_norm": float(id_regret_norm),  # OPTIONAL: Log norm for debug
                        "info_gain": float(info_gain),
                        "ratio": float(ratio)
                    }

                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_hypothesis_ids = h
                else:
                    # Record a high percentage, but don't select it (unless it's the only one).
                    information_ratios[h] = {"regret": float(id_regret_norm), "info_gain": float(info_gain), "ratio": 1000.0}

            # Fallback
            if best_hypothesis_ids:
                selected_hypothesis = best_hypothesis_ids
            else:
                # If all info_gain values are 0, it degenerates into pure exploitation.
                logger.warning("IDS: All candidates have near-zero info gain. Falling back to exploitation.")
                selection_method = "fallback_exploit_min_regret"
                selected_hypothesis = min(regrets, key=regrets.get)  # (Equal to max E[r(h)])

            guidance_message = get_experiment_guidance_prompt(selected_hypothesis, self.task)

        self.submissions.append({
            "source_agent": "experiment_guidance",
            "guidance": guidance_message
        })
        return guidance_message