from typing import (
    Callable,
    List,
    Optional,
)

import logging
from pievo.agents.abstract import Agent
from pievo.group.evolveflow import PiEvo


logger = logging.getLogger(__name__)


class ExperimentAgent(Agent):
    def __init__(
        self,
        strategy: PiEvo | None,
        name: str = "Analysis_Agent",
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):

        super().__init__(
            name=name,
            strategy=strategy,
            system_message=system_message,
            tools=tools,
            **kwargs
        )

    async def get_pievo_guidance(self) -> str:
        """Generate PiEvo guidance for experiment selection"""
        # Extract candidate hypotheses from recent submissions
        candidate_hypotheses = []

        candidates_done = []
        for submission in self.strategy.submissions:
            if submission["source_agent"] == "experiment":
                candidates_done.append(submission["json_data"][0]["candidate"])

        logger.warning(f"Candidates Done for: {candidates_done}")

        for submission in reversed(self.strategy.submissions):
            # Find from the Hypothesis Agent's proposal.
            if submission["source_agent"] == "hypothesis":
                json_data = submission["json_data"]

                # Hypothesis agent may propose many, 1 or 3.
                for key, value in json_data.items():
                    if key.startswith("HYPOTHESIS_") and "candidate" in value:
                        if value["candidate"] not in candidate_hypotheses and value["candidate"] not in candidates_done:
                            candidate_hypotheses.append(value["candidate"])

        if len(candidate_hypotheses) == 0:
            logger.error(f"No candidates Plan (proposed): {candidate_hypotheses} ! ")

        guidance = await self.strategy.get_experiment_guidance(candidate_hypotheses)
        print(f"\n\n[DEBUG] Guidance for `ExperimentAgent`:\n\n=====================\n{guidance}\n=====================\n", flush=True)
        return guidance
