import json
import logging
from typing import (
    Callable,
    List,
    Optional,
)
from pievo.agents.abstract import Agent
from pievo.group.evolveflow import PiEvo

try:
    from pievo.group.manage import pievo_log
except ImportError:

    def pievo_log(content: str, **kwargs) -> None:
        pass


logger = logging.getLogger(__name__)


class ExperimentAgent(Agent):
    def __init__(
        self,
        strategy: PiEvo | None,
        name: str = "Analysis_Agent",
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs,
    ):

        super().__init__(
            name=name,
            strategy=strategy,
            system_message=system_message,
            tools=tools,
            **kwargs,
        )

    async def get_pievo_guidance(self) -> str:
        """Generate PiEvo guidance for experiment selection.

        Collects ALL previously proposed but untested candidate hypotheses from
        the full submission history, ensuring no proposed hypothesis is wasted.
        """
        candidate_hypotheses = []

        # Collect all previously tested candidates
        candidates_done = []
        for submission in self.strategy.submissions:
            if submission["source_agent"] == "experiment":
                jd = submission.get("json_data")
                if isinstance(jd, list) and len(jd) > 0 and isinstance(jd[0], dict):
                    candidate = jd[0].get("candidate")
                    if candidate:
                        candidates_done.append(candidate)
                elif isinstance(jd, dict) and "candidate" in jd:
                    candidates_done.append(jd["candidate"])

        logger.warning(f"Candidates Done (tested): {len(candidates_done)}")

        # Build a set of tested candidate keys for deduplication
        tested_keys = set()
        for c in candidates_done:
            tested_keys.add(
                json.dumps(c, sort_keys=True) if isinstance(c, dict) else str(c)
            )

        # ================================================================
        # Collect ALL untested candidates from EVERY hypothesis submission
        # across the entire history.  This ensures that hypotheses proposed
        # in earlier rounds but never selected for testing are NOT wasted —
        # they are surfaced here as candidates for the current round.
        # ================================================================
        seen_candidate_keys = set()
        for submission in self.strategy.submissions:
            if submission["source_agent"] == "hypothesis":
                json_data = submission.get("json_data")
                if not isinstance(json_data, dict):
                    continue
                for key, value in json_data.items():
                    if (
                        key.startswith("HYPOTHESIS_")
                        and isinstance(value, dict)
                        and "candidate" in value
                    ):
                        candidate = value["candidate"]
                        candidate_key = (
                            json.dumps(candidate, sort_keys=True)
                            if isinstance(candidate, dict)
                            else str(candidate)
                        )
                        if (
                            candidate_key not in tested_keys
                            and candidate_key not in seen_candidate_keys
                        ):
                            candidate_hypotheses.append(candidate)
                            seen_candidate_keys.add(candidate_key)

        # ================================================================
        # Prioritization: put older (long-waiting) untested candidates first
        # so they get tested before newer proposals.
        # ================================================================
        # candidate_hypotheses is already in chronological order because we
        # iterate submissions forward.  Older untested candidates appear first.

        # Diagnostics
        all_proposed = 0
        all_tested = len(candidates_done)
        for submission in self.strategy.submissions:
            if submission["source_agent"] == "hypothesis":
                json_data = submission.get("json_data", {})
                if isinstance(json_data, dict):
                    for key, value in json_data.items():
                        if key.startswith("HYPOTHESIS_") and "candidate" in value:
                            all_proposed += 1

        untested_count = all_proposed - all_tested
        if untested_count < 0:
            untested_count = 0

        if len(candidate_hypotheses) == 0:
            logger.warning(
                f"No untested candidates found. Proposed: {all_proposed}, Tested: {all_tested}. "
                f"The Hypothesis Agent needs to propose more diverse candidates."
            )
            pievo_log(
                f"CRITICAL: No untested candidates available for experiment selection. "
                f"Proposed total: {all_proposed}, Tested: {all_tested}, Untested: {untested_count}. "
                f"This means the Hypothesis Agent is producing repetitive hypotheses.",
                source="experiment_agent",
                tag="candidate_pool_empty",
                level="ERROR",
            )
        else:
            pievo_log(
                f"Candidate pool: {len(candidate_hypotheses)} untested candidates available "
                f"(proposed: {all_proposed}, tested: {all_tested}, untested: {untested_count}).",
                source="experiment_agent",
                tag="candidate_pool",
            )

        guidance = await self.strategy.get_experiment_guidance(candidate_hypotheses)

        # print(f"\n\n[DEBUG] Guidance for `ExperimentAgent`:\n\n=====================\n{guidance}\n=====================\n", flush=True)
        return guidance
