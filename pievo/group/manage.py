import asyncio
import re
import sys
import time
import inspect
import logging
import os
import traceback
from datetime import datetime
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
)
from autogen_core.models import ChatCompletionClient

from pievo.group.notetaker import NoteTaker

logger = logging.getLogger(__name__)

# ── Module-level standard logger for pievo_system.log ──
_pievo_logger: Optional[logging.Logger] = None


def _setup_pievo_logger(output_path: str) -> None:
    """Create a dedicated FileHandler for pievo system logging."""
    global _pievo_logger
    _pievo_logger = logging.getLogger("pievo_system")
    _pievo_logger.setLevel(logging.DEBUG)
    _pievo_logger.propagate = False
    if not _pievo_logger.handlers:
        fh = logging.FileHandler(output_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _pievo_logger.addHandler(fh)


def pievo_log(
    msg: str, *, tag: str = "", source: str = "", level: str = "INFO"
) -> None:
    """Log a message to the pievo_system.log file with caller context."""
    if _pievo_logger is None:
        return
    prefix = f"[{source}] [{tag}]" if source or tag else ""
    formatted = f"{prefix} {msg}" if prefix else msg
    log_method = getattr(_pievo_logger, level.lower(), _pievo_logger.info)
    # Use inspect to attach the caller's line number to the log record
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame else None
        if caller:
            info = inspect.getframeinfo(caller)
            module_name = os.path.splitext(os.path.basename(info.filename))[0]
            log_method(f"[{module_name}:{info.lineno}] {formatted}")
        else:
            log_method(formatted)
    finally:
        del frame


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

        # Track conversation flow
        self.conversation_flow: List[Dict[str, Any]] = []

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

        # Budget tracking
        try:
            self.budget = int(os.environ.get("PIEVO_BUDGET", "0"))
        except (ValueError, TypeError):
            logger.warning(
                f"⚠️ Invalid PIEVO_BUDGET value: {os.environ.get('PIEVO_BUDGET')}. Setting budget to 0 (unlimited)."
            )
            self.budget = 0

        self.experiment_count = 0
        if self.budget > 0:
            logger.info(f"📊 PiEvo Budget Initialized: {self.budget} experiments.")

        self.note_taker = NoteTaker(output_file=note_taker_output_file)

        # System-level plain-text log of every agent's output
        log_dir = os.path.dirname(note_taker_output_file) or "."
        system_log_path = os.path.join(log_dir, "pievo_system.log")
        _setup_pievo_logger(system_log_path)
        logger.info(f"📝 PiEvo system log: {system_log_path}")

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
        # Check budget before proceeding
        if self.budget > 0 and self.experiment_count >= self.budget:
            logger.info(
                f"🏁 Budget reach: {self.experiment_count}/{self.budget} experiments completed. Terminating session."
            )
            return None

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
            pievo_log(
                f"Submission from [{last_speaker}] → switching to next agent",
                source="speaker_selector",
                tag=f"submission:{last_speaker}",
            )

            # Validate that we have proper content before switching
            if hasattr(last_message, "content") and last_message.content:
                logger.debug(
                    f"💬 Last message content preview: {str(last_message.content)[:100]}..."
                )
            else:
                logger.warning(
                    f"⚠️ Last message from {last_speaker} has no content but submission detected"
                )

            # Record the conversation flow entry
            self.conversation_flow.append(
                {
                    "speaker": last_speaker,
                    "turn": self.turn_count,
                    "submission_detected": True,
                    "timestamp": time.time(),
                }
            )

            # Advance to next speaker
            self.current_speaker_name = self._advance_to_next_speaker()
            self.turn_count += 1
            self.last_submission_detected = True

            # Tracking experiment count based on submission from experiment agent
            if "experiment" in last_speaker.lower():
                self.experiment_count += 1
                if self.budget > 0:
                    logger.info(
                        f"🧪 Experiment {self.experiment_count}/{self.budget} completed after submission from {last_speaker}."
                    )
                else:
                    logger.info(
                        f"🧪 Experiment {self.experiment_count} completed after submission from {last_speaker}."
                    )

            logger.info(f"🔄 Switching to: {self.current_speaker_name}")
            pievo_log(
                f"Speaker switch: {last_speaker} → {self.current_speaker_name} (turn={self.turn_count})",
                source="speaker_selector",
                tag="speaker_switch",
            )

            # Log context transfer information for debugging
            self._log_context_transfer(
                last_speaker, self.current_speaker_name, valid_messages
            )

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

            # Record continuation
            self.conversation_flow.append(
                {
                    "speaker": last_speaker,
                    "turn": self.turn_count,
                    "submission_detected": False,
                    "continued": True,
                    "timestamp": time.time(),
                }
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

            # Record tool call start
            self.conversation_flow.append(
                {
                    "speaker": speaker,
                    "turn": self.turn_count,
                    "tool_call_started": True,
                    "submission_detected": False,
                    "timestamp": time.time(),
                }
            )

            return speaker

        # Check if this is a tool execution event
        elif isinstance(message, ToolCallExecutionEvent):
            if self.tool_call_in_progress and self.active_tool_call_agent:
                logger.debug(
                    f"⚙️ Tool execution for {self.active_tool_call_agent} - maintaining speaker lock"
                )

                # Record tool execution
                self.conversation_flow.append(
                    {
                        "speaker": self.active_tool_call_agent,
                        "turn": self.turn_count,
                        "tool_call_executing": True,
                        "submission_detected": False,
                        "timestamp": time.time(),
                    }
                )

                return self.active_tool_call_agent

        # Check if this is the end of a tool call sequence
        elif isinstance(message, ToolCallSummaryMessage):
            if self.tool_call_in_progress and self.active_tool_call_agent:
                logger.debug(
                    f"📊 Tool call completed by {self.active_tool_call_agent} - checking for submission"
                )

                # Record tool call completion
                self.conversation_flow.append(
                    {
                        "speaker": self.active_tool_call_agent,
                        "turn": self.turn_count,
                        "tool_call_completed": True,
                        "submission_detected": False,
                        "timestamp": time.time(),
                    }
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

                    # Record submission
                    self.conversation_flow.append(
                        {
                            "speaker": current_agent,
                            "turn": self.turn_count,
                            "submission_detected": True,
                            "tool_call_submission": True,
                            "timestamp": time.time(),
                        }
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

        # Extract text content from message (handles multimodal)
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = ""
            for c in message.content:
                if isinstance(c, str):
                    content += c
                elif hasattr(c, "text") and c.text:
                    content += c.text
        else:
            content = str(message.content)

        content = content.strip()
        source = getattr(message, "source", "").lower()

        logger.debug(
            f"🔍 Checking submission in message from {source}: '{content[:100]}...'"
        )

        # Special handling for ToolCallSummaryMessage (common for experiment results)
        if isinstance(message, ToolCallSummaryMessage):
            logger.debug(
                f"🔍 Checking tool call summary for submission signals from {source}"
            )

        # Check for submission patterns
        for agent_type, pattern in self.submission_patterns.items():
            if agent_type in source:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    logger.info(
                        f"✅ Detected {agent_type} submission signal from {source}"
                    )
                    return True

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
    def _log_context_transfer(
        from_agent: str, to_agent: str, messages: Sequence[ChatMessage]
    ) -> None:
        """
        Log context transfer information for debugging.

        Args:
            from_agent: The agent that submitted
            to_agent: The agent that will receive context
            messages: The message sequence being transferred
        """
        logger.info(f"🔄 Context transfer: {from_agent} → {to_agent}")

        # Count meaningful messages for context
        text_messages = [
            msg for msg in messages if hasattr(msg, "content") and msg.content
        ]
        tool_messages = [
            msg
            for msg in messages
            if isinstance(
                msg,
                (ToolCallRequestEvent, ToolCallExecutionEvent, ToolCallSummaryMessage),
            )
        ]

        logger.debug(
            f"📜 Context includes: {len(text_messages)} text messages, {len(tool_messages)} tool-related messages"
        )

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
        """
        Override run_stream to add our note-taking, conversation flow tracking,
        and detailed system logging of every agent output, tool call, and event.
        """
        task_preview = str(task)[:200] if task else "N/A"
        pievo_log(f"task={task_preview}", source="group_chat", tag="SESSION_START")
        logger.info("🚀 Starting SubmissionBasedGroupChat stream...")

        try:
            # Use the parent's run_stream but add our tracking
            async for message in super().run_stream(
                task=task, cancellation_token=cancellation_token
            ):
                try:
                    source = getattr(message, "source", "unknown")
                    msg_type = type(message).__name__

                    # ── System log: faithfully record every agent output ──
                    if hasattr(message, "content") and message.content:
                        # Extract text content for logging (handles multimodal)
                        if isinstance(message.content, str):
                            content_str = message.content
                        elif isinstance(message.content, list):
                            content_str = ""
                            for c in message.content:
                                if isinstance(c, str):
                                    content_str += c
                                elif hasattr(c, "text") and c.text:
                                    content_str += c.text
                        else:
                            content_str = str(message.content)

                        content_str = content_str.strip()

                        if isinstance(message, TextMessage):
                            if content_str:
                                pievo_log(
                                    content_str, source=source, tag="agent_output"
                                )

                        elif isinstance(message, ToolCallRequestEvent):
                            for call in message.content:
                                call_name = getattr(call, "name", "unknown")
                                call_args = getattr(call, "arguments", "")
                                pievo_log(
                                    f"TOOL CALL: {call_name}\nArguments: {call_args}",
                                    source=source,
                                    tag="tool_request",
                                )

                        elif isinstance(message, ToolCallExecutionEvent):
                            for result in message.content:
                                # result.content can also be multimodal if tool returns Image
                                has_image = False
                                if isinstance(result.content, str):
                                    resp_content = result.content
                                elif isinstance(result.content, list):
                                    resp_content = ""
                                    for c in result.content:
                                        if isinstance(c, str):
                                            resp_content += c
                                        # Use type name check as well for robustness
                                        elif (
                                            hasattr(c, "data")
                                            or hasattr(c, "url")
                                            or type(c).__name__ == "Image"
                                        ):
                                            has_image = True
                                            resp_content += "[IMAGE ATTACHED] "
                                        elif hasattr(c, "text"):
                                            resp_content += c.text
                                        else:
                                            resp_content += f"[{type(c).__name__}] "
                                else:
                                    resp_content = str(result.content)

                                is_error = getattr(result, "is_error", False)
                                img_tag = " [WITH IMAGE]" if has_image else ""
                                pievo_log(
                                    f"TOOL RESULT (error={is_error}){img_tag}: {resp_content[:500]}",
                                    source=source,
                                    tag="tool_result",
                                )

                        elif isinstance(message, ToolCallSummaryMessage):
                            pievo_log(
                                content_str,
                                source=source,
                                tag="tool_summary",
                            )

                        elif isinstance(message, StopMessage):
                            pievo_log(
                                content_str,
                                source=source,
                                tag="stop",
                            )
                        else:
                            # Fallback: log any other message type that has content
                            pievo_log(
                                content_str[:500],
                                source=source,
                                tag=f"message:{msg_type}",
                            )

                    elif isinstance(message, TaskResult):
                        stop_reason = getattr(message, "stop_reason", "none")
                        pievo_log(
                            f"TaskResult: stop_reason={stop_reason}",
                            source=source,
                            tag="task_result",
                        )

                    # Record message in note taker (with error handling)
                    if hasattr(message, "content") and not isinstance(
                        message, ModelClientStreamingChunkEvent
                    ):
                        try:
                            self.note_taker.record_message(message)
                        except Exception as note_error:
                            logger.warning(f"⚠️ Note taker error: {note_error}")

                        self.note_taker.save()

                    # Yield the message (preserve all messages for context continuity)
                    yield message

                    # Save periodically and handle final results
                    if isinstance(message, TaskResult):
                        try:
                            self.note_taker.save()
                            self._save_conversation_flow()
                            pievo_log(
                                "All turns completed",
                                source="group_chat",
                                tag="session_end",
                            )
                            logger.info(
                                f"💾 Final conversation saved. Total turns: {self.turn_count}"
                            )
                        except Exception as save_error:
                            logger.warning(f"⚠️ Error saving final state: {save_error}")

                except Exception as message_error:
                    pievo_log(
                        f"Error processing message: {message_error}\n{traceback.format_exc()}",
                        source="group_chat",
                        tag="error",
                        level="ERROR",
                    )
                    logger.warning(f"⚠️ Error processing message: {message_error}")
                    yield message

        except asyncio.CancelledError:
            pievo_log(
                "Stream cancelled", source="group_chat", tag="cancelled", level="WARN"
            )
            logger.info("🛑 Stream was cancelled")
            if self.tool_call_in_progress:
                self.tool_call_in_progress = False
                self.active_tool_call_agent = None

            try:
                self.note_taker.save()
                self._save_conversation_flow()
                pievo_log(
                    "State saved before cancellation",
                    source="group_chat",
                    tag="cleanup",
                )
                logger.info("💾 Saved state before cancellation")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Error during cleanup: {cleanup_error}")

            raise

        except Exception as stream_error:
            pievo_log(
                f"Critical error in stream: {stream_error}\n{traceback.format_exc()}",
                source="group_chat",
                tag="critical_error",
                level="ERROR",
            )
            logger.error(f"❌ Critical error in stream: {stream_error}")

            if self.tool_call_in_progress:
                self.tool_call_in_progress = False
                self.active_tool_call_agent = None

            try:
                self.note_taker.save()
                self._save_conversation_flow()
                logger.info("💾 Saved partial results after error")
            except Exception as save_error:
                logger.warning(f"⚠️ Could not save partial results: {save_error}")

            error_result = TaskResult(
                messages=[
                    TextMessage(
                        content=f"Stream error: {stream_error}", source="system"
                    )
                ],
                stop_reason=f"error: {stream_error}",
            )
            yield error_result

    def _save_conversation_flow(self):
        """Save the conversation flow analysis."""
        try:
            import json
            import os

            flow_file = self.note_taker.output_file.replace(".json", "_flow.json")

            # Calculate statistics
            speaker_stats = {}

            for entry in self.conversation_flow:
                speaker = entry["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "total_turns": 0,
                        "submissions": 0,
                        "continuations": 0,
                        "tool_calls_started": 0,
                        "tool_calls_completed": 0,
                        "tool_call_submissions": 0,
                    }

                speaker_stats[speaker]["total_turns"] += 1

                if entry["submission_detected"]:
                    speaker_stats[speaker]["submissions"] += 1
                elif entry.get("continued", False):
                    speaker_stats[speaker]["continuations"] += 1

                # Track tool call statistics
                if entry.get("tool_call_started", False):
                    speaker_stats[speaker]["tool_calls_started"] += 1
                if entry.get("tool_call_completed", False):
                    speaker_stats[speaker]["tool_calls_completed"] += 1
                if entry.get("tool_call_submission", False):
                    speaker_stats[speaker]["tool_call_submissions"] += 1

            # Calculate overall tool call statistics
            total_tool_calls = sum(
                stats["tool_calls_started"] for stats in speaker_stats.values()
            )
            completed_tool_calls = sum(
                stats["tool_calls_completed"] for stats in speaker_stats.values()
            )

            flow_data = {
                "conversation_flow": self.conversation_flow,
                "speaker_statistics": speaker_stats,
                "agent_rotation_order": self.agent_rotation_order,
                "total_turns": self.turn_count,
                "tool_call_summary": {
                    "total_tool_calls_started": total_tool_calls,
                    "total_tool_calls_completed": completed_tool_calls,
                    "tool_call_success_rate": (
                        completed_tool_calls / total_tool_calls
                        if total_tool_calls > 0
                        else 0
                    ),
                    "current_tool_call_in_progress": self.tool_call_in_progress,
                    "active_tool_call_agent": self.active_tool_call_agent,
                },
                "summary": {
                    "total_speakers": len(speaker_stats),
                    "total_conversation_entries": len(self.conversation_flow),
                    "consecutive_speaking_enabled": True,
                    "tool_call_management_enabled": True,
                },
            }

            with open(flow_file, "w") as f:
                json.dump(flow_data, f, indent=2)

            logger.info(f"💾 Conversation flow saved to: {flow_file}")
            logger.info(
                f"📊 Tool calls: {total_tool_calls} started, {completed_tool_calls} completed"
            )

        except Exception as e:
            logger.warning(f"⚠️ Error saving conversation flow: {e}")
