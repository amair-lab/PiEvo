import logging
from typing import Callable, List, Optional

from autogen_core import EVENT_LOGGER_NAME
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from watchfiles import awatch

from pievo.group.evolveflow import PiEvo
from pievo.agents.abstract import Agent

event_logger = logging.getLogger(EVENT_LOGGER_NAME)


class PrincipleAgent(Agent):
    def __init__(
        self,
        name: str = "PrincipleAgent",
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        model_client: OpenAIChatCompletionClient | ChatCompletionClient = None,
        strategy: PiEvo | None = None,
        **kwargs,
    ):
        default_system_message = "You plan the task by decoupling and assignment. "

        super().__init__(
            name=name,
            strategy=strategy,
            model_client=model_client,
            system_message=system_message or default_system_message,
            tools=tools,
            **kwargs,
        )

        self.strategy = strategy

    async def get_pievo_guidance(self) -> str:
        """Generate PiEvo guidance for principle generation"""
        guidance = await self.strategy.get_principle_guidance()

        # Update global round: round_by_PHE_order_before_P
        # ONLY +1 HERE!
        self.strategy.round_by_PHE_order_before_P += 1
        return guidance
