from abc import ABC
from typing import Callable, List, Optional

from watchfiles import awatch

from pievo.agents.abstract import Agent
from pievo.group.evolveflow import PiEvo


class HypothesisAgent(Agent):
    def __init__(
        self,
        strategy: PiEvo | None,
        name: str = "Analysis_Agent",
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs,
    ):
        default_system_message = """You are a Hypothesis Agent specialized in formulating, refining, and testing 
        scientific hypotheses about Large Language Models. Your expertise lies in connecting theoretical frameworks 
        with empirical observations, identifying potential causal relationships, and proposing testable predictions.
        You excel at critical thinking, maintaining scientific rigor, and adjusting hypotheses based on new evidence.
        Focus on clarity, falsifiability, and scientific value when generating hypotheses."""

        super().__init__(
            strategy=strategy,
            name=name,
            system_message=system_message or default_system_message,
            tools=tools,
            **kwargs,
        )

    async def get_pievo_guidance(self) -> str:
        """Generate PiEvo guidance for hypothesis generation"""

        guidance = await self.strategy.get_hypothesis_guidance()

        # print(f"\n\n[DEBUG] Guidance for `HypothesisAgent`:\n\n=====================\n{guidance}\n=====================\n", flush=True)
        return guidance
