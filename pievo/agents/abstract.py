import abc
import asyncio
import logging
import uuid
from typing import (
    AsyncGenerator,
    Callable,
    List,
    Sequence,
    Tuple,
    Optional,
    Union,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    HandoffMessage,
    AgentEvent,
    ChatMessage,
    ThoughtEvent,
    BaseAgentEvent,
    BaseChatMessage,
)
from autogen_core import FunctionCall, CancellationToken, EVENT_LOGGER_NAME
from autogen_core.models import (
    UserMessage,
    CreateResult,
    AssistantMessage,
    RequestUsage,
    SystemMessage,
    ChatCompletionClient,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pievo.group.evolveflow import PiEvo

event_logger = logging.getLogger(EVENT_LOGGER_NAME)


class Agent(AssistantAgent):
    """Agent responsible for analyzing data and results."""

    def __init__(
        self,
        strategy: PiEvo,
        name: str = "Agent",
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        model_client: OpenAIChatCompletionClient | ChatCompletionClient = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            tools=tools,
            **kwargs,
        )

        # is equal to OpenAIChatCompletionClient, using `.create(...)` to get LLM response.
        self.llm_client = model_client
        self.strategy = strategy
        self.processed_messages = set()  # Track which messages have been processed

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        system_messages = self._system_messages
        workbench = self._workbench
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = False
        max_tool_iterations = self._max_tool_iterations
        tool_call_summary_format = self._tool_call_summary_format
        tool_call_summary_formatter = self._tool_call_summary_formatter
        output_content_type = self._output_content_type

        # STEP 1: Add new user/handoff messages to the model context
        # Validate and filter messages before adding to context
        from pievo.group.manage import pievo_log

        pievo_log(
            f"Processing {len(messages)} messages for {agent_name}",
            source=f"Agent:{agent_name}",
            tag="on_messages_start",
        )

        # Check if this agent supports vision
        has_vision = getattr(self._model_client, "model_info", {}).get("vision", False)

        valid_messages = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content is not None:
                # Vision filtering logic
                content = msg.content
                if not has_vision and isinstance(content, list):
                    # Filter out Image parts for non-vision agents
                    filtered_content = [
                        c
                        for c in content
                        if not (
                            hasattr(c, "data")
                            or hasattr(c, "url")
                            or type(c).__name__ == "Image"
                        )
                    ]
                    if len(filtered_content) == 1 and isinstance(
                        filtered_content[0], str
                    ):
                        content = filtered_content[0]
                    else:
                        content = filtered_content

                    pievo_log(
                        f"Vision filtering: Stripped image parts for non-vision agent {agent_name}",
                        source=f"Agent:{agent_name}",
                        tag="vision_filter",
                    )

                # Create a new message object if content was modified, or use original
                # Note: To avoid mutating the original message which might be used by other agents,
                # we should ideally pass a modified version to _add_messages_to_context
                # AssistantAgent._add_messages_to_context typically handles ChatMessage

                # Check for multimodal status for logging
                content_type = type(content).__name__
                is_multimodal = isinstance(content, list)
                parts_info = ""
                if is_multimodal:
                    parts_info = f" parts: {[type(c).__name__ for c in content]}"

                pievo_log(
                    f"Adding valid message from {getattr(msg, 'source', 'unknown')}. Type: {content_type}, Multi: {is_multimodal}{parts_info}",
                    source=f"Agent:{agent_name}",
                    tag="context_update",
                )

                # If we modified the content, we need to pass a message with modified content
                if content is not msg.content:
                    # Create a shallow copy or a new message of the same type with new content
                    import copy

                    msg_copy = copy.copy(msg)
                    msg_copy.content = content
                    valid_messages.append(msg_copy)
                else:
                    valid_messages.append(msg)
            else:
                pievo_log(
                    f"Skipping null content message from {getattr(msg, 'source', 'unknown')}",
                    source=f"Agent:{agent_name}",
                    tag="context_skip",
                    level="WARNING",
                )

        if valid_messages:
            await self._add_messages_to_context(
                model_context=model_context,
                messages=valid_messages,
            )
        else:
            event_logger.warning(
                f"⚠️ No valid messages to add to context for {agent_name}"
            )

        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        for event_msg in await self._update_model_context_with_memory(
            memory=memory,
            model_context=model_context,
            agent_name=agent_name,
        ):
            inner_messages.append(event_msg)
            yield event_msg

        # STEP 3: Generate a message ID for correlation between streaming chunks and final message
        message_id = str(uuid.uuid4())
        model_result = None
        system_messages = list(system_messages)

        if self.strategy:
            self.strategy.gather_submission_from_message(messages)

        if not self.strategy.off_pievo:
            await model_context.add_message(
                UserMessage(content=await self.get_pievo_guidance(), source="user")
            )
        else:
            event_logger.warning(
                f"⚠️ Currently, you are running without PiEvo for agent `{agent_name}`, only workable in ablation period. "
            )

        # =================== Handle the different reasoning modes ===================

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                async for inference_output in self._call_llm(
                    model_client=model_client,
                    model_client_stream=model_client_stream,
                    system_messages=system_messages,
                    model_context=model_context,
                    workbench=workbench,
                    handoff_tools=handoff_tools,
                    agent_name=agent_name,
                    cancellation_token=cancellation_token,
                    output_content_type=output_content_type,
                    message_id=message_id,
                ):
                    if isinstance(inference_output, CreateResult):
                        model_result = inference_output
                        if not model_result.content:
                            event_logger.warning(
                                f"⚠️ LLM returned empty content for {agent_name}"
                            )
                    else:
                        # Streaming chunk event
                        yield inference_output
                break  # success — exit retry loop
            except Exception as e:
                event_logger.error(
                    f"❌ LLM call failed for {agent_name} (attempt {attempt}/{max_attempts}): {e}"
                )
                if attempt < max_attempts:
                    wait = 5 * attempt  # 5s, 10s, 15s, 20s
                    event_logger.warning(f"⏳ Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    model_result = CreateResult(
                        finish_reason="stop",
                        content=f"[Meta Class of Agent] Error in LLM processing. Please check the history management or model call. Error: {str(e)}",
                        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                        cached=False,
                        logprobs=None,
                        thought=None,
                    )
                    raise ConnectionError(e)

        # --- If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(
                content=model_result.thought, source=agent_name
            )
            yield thought_event
            inner_messages.append(thought_event)

        # Add the assistant message to the model context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # STEP 5: Process the model output
        async for output_event in self._process_model_result(
            model_result=model_result,
            inner_messages=inner_messages,
            cancellation_token=cancellation_token,
            agent_name=agent_name,
            system_messages=system_messages,
            model_context=model_context,
            workbench=workbench,
            handoff_tools=handoff_tools,
            handoffs=handoffs,
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            max_tool_iterations=max_tool_iterations,
            tool_call_summary_format=tool_call_summary_format,
            tool_call_summary_formatter=tool_call_summary_formatter,
            output_content_type=output_content_type,
            message_id=message_id,
            format_string=self._output_content_type_format,
        ):
            yield output_event

    # This will be called only if self.strategy is True (or an PiEvo class).
    async def get_pievo_guidance(self) -> str:
        """Generate PiEvo guidance - should be implemented by subclasses"""
        raise NotImplementedError("No specific guidance available.")
