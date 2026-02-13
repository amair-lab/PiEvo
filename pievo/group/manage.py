import asyncio
import re
import time
import logging
from typing import List, Optional, Sequence, AsyncGenerator, Dict, Any, Tuple, Callable
from autogen_agentchat.base import ChatAgent, TerminationCondition, TaskResult
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    TextMessage,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    BaseAgentEvent,
    StopMessage,
    MessageFactory,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams._group_chat._selector_group_chat import (
    SelectorGroupChat,
    SelectorGroupChatConfig,
    SelectorFuncType,
    CandidateFuncType,
)
from autogen_core import (
    AgentRuntime,
    CancellationToken,
    AgentId,
    SingleThreadedAgentRuntime,
)
from autogen_core.models import ChatCompletionClient, UserMessage


logger = logging.getLogger(__name__)


class SubmissionBasedGroupChat(SelectorGroupChat):
    """
    Custom Group Chat that allows agents to speak multiple times until they submit.
    Switches to the next agent when a SUBMIT signal is detected.
    Follows a fixed rotation: Planner -> Hypothesis -> Experiment -> Planner...
    """

    component_config_schema = SelectorGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.SubmissionBasedGroupChat"

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        note_taker_output_file: str = "chat_history.json",
        agent_rotation_order: List[str] = None,
        submission_patterns: Dict[str, str] = None,
        selector_prompt: str = "Select next speaker based on submission signals and rotation order.",
        allow_repeated_speaker: bool = True,
        max_selector_attempts: int = 3,
        selector_func: Optional[SelectorFuncType] = None,
        candidate_func: Optional[CandidateFuncType] = None,
        custom_message_types: (
            List[type[BaseAgentEvent | BaseChatMessage]] | None
        ) = None,
        emit_team_events: bool = False,
        model_client_streaming: bool = False,
    ):
        """
        Initialize the submission-based group chat.

        Args:
            participants: List of chat agents
            model_client: Chat completion client for speaker selection
            termination_condition: When to stop the conversation
            max_turns: Maximum number of turns
            runtime: Agent runtime to use
            note_taker_output_file: File to save conversation history
            agent_rotation_order: Order of agent rotation (default: ["planner", "hypothesis", "experiment"])
            submission_patterns: Patterns to detect submissions for each agent type
            selector_prompt: Prompt for speaker selection (inherited from SelectorGroupChat)
            allow_repeated_speaker: Whether to allow repeated speakers (must be True for consecutive speaking)
            max_selector_attempts: Maximum attempts for speaker selection
            selector_func: Custom selector function (we'll create our own)
            candidate_func: Custom candidate function
            custom_message_types: Custom message types
            emit_team_events: Whether to emit team events
            model_client_streaming: Whether to use streaming
        """

        self.agent_rotation_order = agent_rotation_order
        self.current_speaker_index = 0  # Start with first agent (planner)
        self.current_speaker_name = None
        self.turn_count = 0
        self.last_submission_detected = False

        # Tool call state tracking
        self.active_tool_call_agent = (
            None  # Track which agent is in a tool call sequence
        )
        self.tool_call_in_progress = (
            False  # Whether we're in the middle of a tool call sequence
        )

        # Submission detection patterns
        self.submission_patterns = submission_patterns

        # Agent mapping
        self.agent_map = {agent.name: agent for agent in participants}

        # Create our custom selector function
        def submission_based_selector(
            messages: Sequence[AgentEvent | ChatMessage],
        ) -> str | None:
            return self._select_next_speaker(messages)

        # Initialize parent SelectorGroupChat with our custom selector
        super().__init__(
            participants=participants,
            model_client=model_client,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=allow_repeated_speaker,  # Must be True for consecutive speaking
            max_selector_attempts=max_selector_attempts,
            selector_func=submission_based_selector,  # Use our custom function
            candidate_func=candidate_func,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
            model_client_streaming=model_client_streaming,
        )


    def _select_next_speaker(
        self, messages: Sequence[AgentEvent | ChatMessage]
    ) -> str | None:
        """
        Custom speaker selection logic that handles submission-based switching and tool call sequences.
        This method is called by the SelectorGroupChat framework.

        Args:
            messages: Sequence of messages in the conversation

        Returns:
            str: Name of the next speaker, or None to terminate
        """
        if not messages:
            # Start with the first agent in rotation
            self.current_speaker_name = self._get_current_speaker_name()
            logger.info(f"🚀 Starting conversation with: {self.current_speaker_name}")
            return self.current_speaker_name

        # Process messages but preserve context - be more lenient with filtering
        valid_messages = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                # Accept TextMessages even with minimal content to preserve context flow
                if hasattr(msg, "content") and msg.content is not None:
                    # Accept any non-None content, including empty strings that might carry context
                    valid_messages.append(msg)
                    if str(msg.content).strip():
                        logger.debug(
                            f"✅ Including message with content from {getattr(msg, 'source', 'unknown')}"
                        )
                    else:
                        logger.debug(
                            f"✅ Including empty content message from {getattr(msg, 'source', 'unknown')} for context continuity"
                        )
                else:
                    logger.debug(
                        f"⚠️ Skipping message with None content from {getattr(msg, 'source', 'unknown')}"
                    )
            elif isinstance(msg, ModelClientStreamingChunkEvent):
                # Include streaming chunk events as they may contain JSON blocks from principle_agent
                if hasattr(msg, "content") and msg.content is not None:
                    valid_messages.append(msg)
                    logger.debug(
                        f"✅ Including streaming chunk from {getattr(msg, 'source', 'unknown')}"
                    )
            else:
                # Always include non-TextMessage types (events, tool calls, etc.)
                valid_messages.append(msg)

        if not valid_messages:
            # No valid messages, start with first agent
            self.current_speaker_name = self._get_current_speaker_name()
            logger.info(
                f"🚀 No valid messages, starting with the first speaker: {self.current_speaker_name}"
            )
            return self.current_speaker_name

        # Get the last valid message
        last_message = valid_messages[-1]
        last_speaker = getattr(last_message, "source", "")

        # Handle user messages
        if last_speaker in ["user", "user_proxy"]:
            self.current_speaker_name = self._get_current_speaker_name()
            self.tool_call_in_progress = False  # Reset tool call state
            self.active_tool_call_agent = None
            logger.debug(f"👤 User spoke, starting with: {self.current_speaker_name}")
            return self.current_speaker_name

        # Handle tool call sequences
        tool_call_result = self._handle_tool_call_sequence(last_message, last_speaker)
        if tool_call_result is not None:
            return tool_call_result

        # Check if we should switch speakers based on submission signal (only if not in tool call)
        if not self.tool_call_in_progress and self._detect_submission_signal(
            last_message
        ):
            logger.info(f"✅ Submission detected from {last_speaker}")

            # Validate that we have proper content before switching
            if hasattr(last_message, "content") and last_message.content:
                logger.debug(
                    f"💬 Last message content preview: {str(last_message.content)[:100]}..."
                )
            else:
                logger.warning(
                    f"⚠️ Last message from {last_speaker} has no content but submission detected"
                )

            # Advance to next speaker
            self.current_speaker_name = self._advance_to_next_speaker()
            self.turn_count += 1
            self.last_submission_detected = True

            logger.info(f"🔄 Switching to: {self.current_speaker_name}")


            # Verify the target agent exists
            if self.current_speaker_name not in self.agent_map:
                logger.error(
                    f"❌ Target agent '{self.current_speaker_name}' not found in agent map"
                )
                # Fallback to first available agent
                available_agents = list(self.agent_map.keys())
                if available_agents:
                    self.current_speaker_name = available_agents[0]
                    logger.warning(f"🔄 Falling back to: {self.current_speaker_name}")

            return self.current_speaker_name

        # No submission detected and not in tool call - continue with same speaker (if consecutive speaking is allowed)
        if last_speaker and last_speaker in self.agent_map:
            logger.debug(
                f"➡️ {last_speaker} continues speaking (no submission detected)"
            )

            self.current_speaker_name = last_speaker
            self.last_submission_detected = False
            return last_speaker

        # Fallback: use rotation order
        self.current_speaker_name = self._get_current_speaker_name()
        logger.debug(f"🔄 Fallback to rotation: {self.current_speaker_name}")
        return self.current_speaker_name

    def _handle_tool_call_sequence(
        self, message: ChatMessage, speaker: str
    ) -> str | None:
        """
        Handle tool call sequences to ensure they complete without speaker interruption.

        Args:
            message: The current message
            speaker: The speaker who sent the message

        Returns:
            str: Speaker name if tool call sequence should continue, None if normal logic should apply
        """
        # Check if this is the start of a tool call sequence
        if isinstance(message, ToolCallRequestEvent):
            logger.debug(
                f"🔧 Tool call started by {speaker} - maintaining speaker lock"
            )
            self.tool_call_in_progress = True
            self.active_tool_call_agent = speaker
            self.current_speaker_name = speaker

            return speaker

        # Check if this is a tool execution event
        elif isinstance(message, ToolCallExecutionEvent):
            if self.tool_call_in_progress and self.active_tool_call_agent:
                logger.debug(
                    f"⚙️ Tool execution for {self.active_tool_call_agent} - maintaining speaker lock"
                )

                return self.active_tool_call_agent

        # Check if this is the end of a tool call sequence
        elif isinstance(message, ToolCallSummaryMessage):
            if self.tool_call_in_progress and self.active_tool_call_agent:
                logger.debug(
                    f"📊 Tool call completed by {self.active_tool_call_agent} - checking for submission"
                )

                # Tool call sequence is complete, reset state
                current_agent = self.active_tool_call_agent
                self.tool_call_in_progress = False
                self.active_tool_call_agent = None

                # Now check for submission signals in the tool summary
                if self._detect_submission_signal(message):
                    logger.info(
                        f"✅ Submission detected in tool summary from {current_agent}"
                    )

                    # Advance to next speaker
                    self.current_speaker_name = self._advance_to_next_speaker()
                    self.turn_count += 1
                    self.last_submission_detected = True

                    logger.info(f"🔄 Switching to: {self.current_speaker_name}")
                    return self.current_speaker_name
                else:
                    # Tool call completed but no submission - agent can continue
                    logger.debug(
                        f"➡️ {current_agent} can continue after tool call (no submission)"
                    )
                    self.current_speaker_name = current_agent
                    return current_agent

        # If we're in a tool call sequence but this message doesn't match expected types
        elif self.tool_call_in_progress and self.active_tool_call_agent:
            logger.debug(
                f"🔧 Tool call sequence in progress - maintaining {self.active_tool_call_agent}"
            )
            return self.active_tool_call_agent

        # Not a tool call related message or sequence
        return None

    def _get_current_speaker_name(self) -> str:
        """Get the name of the agent who should speak next based on rotation."""
        if self.current_speaker_index >= len(self.agent_rotation_order):
            self.current_speaker_index = 0

        target_role = self.agent_rotation_order[self.current_speaker_index]

        # Find agent with matching role in name
        for agent_name in self.agent_map.keys():
            if target_role.lower() in agent_name.lower():
                return agent_name

        # Fallback: return first available agent
        return list(self.agent_map.keys())[0] if self.agent_map else ""

    def _detect_submission_signal(self, message: ChatMessage) -> bool:
        """
        Detect if the message contains a submission signal.
        Enhanced to handle different message types including tool call summaries.

        Args:
            message: The message to check

        Returns:
            bool: True if submission signal detected
        """
        # Use the validation helper
        if not self._is_valid_message(message):
            logger.debug(
                f"🔍 Skipping submission detection for invalid message from {getattr(message, 'source', 'unknown')}"
            )
            return False

        content = str(message.content).strip()
        source = getattr(message, "source", "").lower()

        logger.debug(
            f"🔍 Checking submission in message from {source}: '{content[:100]}...'"
        )

        # Special handling for ToolCallSummaryMessage (common for experiment results)
        if isinstance(message, ToolCallSummaryMessage):
            logger.debug(
                f"🔍 Checking tool call summary for submission signals from {source}"
            )
            # Tool call summaries often contain experiment results and submission signals
            # Check content more thoroughly for these cases

        # Check for submission patterns
        for agent_type, pattern in self.submission_patterns.items():
            if agent_type in source:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    logger.info(
                        f"✅ Detected {agent_type} submission signal from {source}"
                    )
                    logger.debug(
                        f"🔍 Submission pattern matched: '{pattern}' in content: '{content[:200]}...'"
                    )
                    return True
                else:
                    logger.debug(
                        f"🔍 Pattern '{pattern}' not found in {agent_type} content from {source}"
                    )
            else:
                logger.debug(f"🔍 Agent type '{agent_type}' not in source '{source}'")

        # Also check for generic submission signals
        generic_signals = [
            "SUBMISSION",
        ]

        for signal in generic_signals:
            if signal in content.upper():
                logger.info(
                    f"✅ Detected generic submission signal '{signal}' from {source}"
                )
                return True

        if "experiment" in source and isinstance(message, ToolCallSummaryMessage):
            # Check if the content looks like a completed experiment result
            result_indicators = ["SUBMISSION"]
            for indicator in result_indicators:
                if indicator.lower() in content.lower():
                    logger.debug(
                        f"🧪 Detected experiment completion indicator '{indicator}' from {source}"
                    )
                    return True

        return False

    def _advance_to_next_speaker(self) -> str:
        self.current_speaker_index = (self.current_speaker_index + 1) % len(
            self.agent_rotation_order
        )
        next_speaker = self._get_current_speaker_name()

        logger.debug(
            f"🔄 Advancing to next speaker: {next_speaker} (index: {self.current_speaker_index})"
        )
        return next_speaker

    @staticmethod
    def _is_valid_message(message: ChatMessage) -> bool:
        """
        Check if a message has valid content that should be processed.
        Relaxed validation to preserve context transfer.

        Args:
            message: The message to validate

        Returns:
            bool: True if message is valid and should be processed
        """
        if not hasattr(message, "content"):
            return False

        if message.content is None:
            return False

        # More permissive: allow empty strings for context continuity
        # Only reject if there's truly no content at all
        return True

    async def run_stream(
        self,
        *,
        task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
        output_task_messages: bool = True,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        """Minimalist stream runner."""
        async for message in super().run_stream(
            task=task,
            cancellation_token=cancellation_token,
        ):
            yield message

