import abc
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
        reflect_on_tool_use = True
        max_tool_iterations = self._max_tool_iterations
        tool_call_summary_format = self._tool_call_summary_format
        tool_call_summary_formatter = self._tool_call_summary_formatter
        output_content_type = self._output_content_type

        # STEP 1: Add new user/handoff messages to the model context
        # Validate and filter messages before adding to context
        valid_messages = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                valid_messages.append(msg)
                event_logger.debug(
                    f"📨 Adding valid message from {getattr(msg, 'source', 'unknown')}: {str(msg.content)[:100]}..."
                )
            else:
                event_logger.warning(
                    f"⚠️ Skipping empty message from {getattr(msg, 'source', 'unknown')}"
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


        # [STEP OMNI] FOR **ALL**: Listen for all messages for all baselines & PiFlow.
        # NOTE: =============== For PiEvo computation (and yield the suggestion / strategy) ===============
        #       This block does not conduct the internal mechanism of PiEvo, but be opened for all cases ON-OFF pievo,
        #       the reason to keep it is only to collect chatting history of agents structurly. 
        if self.strategy:
            self.strategy.gather_submission_from_message(messages)
        

        # =================== Handle the different reasoning modes =================== 
        
        # NOTE: This code block serve as a prompt injection implicitly, where it operates on the Agent's context.
        #       But it does not appear to the main context of the group chatting.
        #       So it is good enough for serving as a steering support.
        #       Create a temporary context with the flow message for this inference. 
        if not self.strategy.off_pievo:
            await model_context.add_message(
                UserMessage(content=await self.get_pievo_guidance(), source="user")
            )
        else:
            event_logger.warning(
                f"⚠️ Currently, you are running without PiEvo for agent `{agent_name}`, only workable in ablation period. "
            )
        
        # =================== Handle the different reasoning modes =================== 


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
        except Exception as e:
            event_logger.error(f"❌ LLM call failed for {agent_name}: {e}")

            # Create a fallback result
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
