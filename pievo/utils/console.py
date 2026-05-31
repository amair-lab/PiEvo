import asyncio
import json
import os
import sys
import time
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    TypeVar,
    Union,
)

from autogen_agentchat.teams._group_chat._events import GroupChatTermination
from autogen_ext.ui._rich_console import _image_to_iterm
from colorama import Fore, Style, init

from autogen_agentchat.ui import UserInputManager
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage

from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    UserInputRequestedEvent,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    ToolCallSummaryMessage,
)


def _is_running_in_iterm() -> bool:
    return os.getenv("TERM_PROGRAM") == "iTerm.app"


def _is_output_a_tty() -> bool:
    return sys.stdout.isatty()


import re


def _truncate_base64(text: str) -> str:
    """Find and truncate base64 strings and large numerical arrays in text."""
    if not isinstance(text, str):
        return text

    # 1. Truncate standard data URI base64
    def replace_b64(match):
        prefix = match.group(1) or ""
        b64_content = match.group(2)
        if len(b64_content) > 50:
            return f"{prefix}{b64_content[:10]}...[truncated base64, len={len(b64_content)}]{b64_content[-10:]}"
        return match.group(0)

    text = re.sub(r"(data:[^;]+;base64,)([A-Za-z0-9+/=]{20,})", replace_b64, text)

    # 2. Truncate standalone long base64-like or large binary sequences
    def replace_standalone(match):
        s = match.group(0)
        if len(s) > 80:
            return f"{s[:15]}...[truncated sequence, len={len(s)}]{s[-15:]}"
        return s

    text = re.sub(r"[A-Za-z0-9+/=]{80,}", replace_standalone, text)

    # 3. Truncate large numerical arrays (sequential data for plotting)
    # This improved regex handles scientific notation and complex floats
    # Match sequences like [0.123, 1.5e-10, ... 10+ times]
    def replace_array(match):
        array_str = match.group(0)
        # Count elements by looking for numbers separated by commas
        # This is a heuristic to avoid over-truncating tiny lists
        if array_str.count(",") > 8:
            # Simple approach: keep the brackets but truncate the inner text
            # We want to show it's an array but hide the bulk
            # Find the first few characters after [
            start = array_str[:30]
            # Find the last few characters before ]
            end = array_str[-30:]
            return f"{start} ... [truncated sequential data] ... {end}"
        return array_str

    # Match JSON-style arrays of numbers (including scientific notation)
    # We look for a pattern that matches a list start '[' followed by many floats/ints and ending with ']'
    # Using a slightly more aggressive matching for large blocks of numbers
    text = re.sub(
        r"\[\s*-?\d+\.?\d*[eE]?[+-]?\d*(?:\s*,\s*-?\d+\.?\d*[eE]?[+-]?\d*){10,}\s*\]",
        replace_array,
        text,
    )

    return text


def _message_to_str(
    message: AgentEvent | ChatMessage, *, render_image_iterm: bool = False
) -> str:

    if isinstance(message, MultiModalMessage):
        result: List[str] = []
        for c in message.content:
            if isinstance(c, str):
                result.append(_truncate_base64(c))
            else:
                if render_image_iterm:
                    result.append(_image_to_iterm(c))
                else:
                    result.append("<image>")
        return "\n".join(result)
    elif isinstance(message, ToolCallRequestEvent) and len(message.content) >= 1:
        return f"🔧(id={message.content[0].id}) \n{message.content[0].name}({message.content[0].arguments})"
    elif isinstance(message, ToolCallExecutionEvent) and len(message.content) >= 1:
        try:
            # Try to parse and format JSON, then truncate base64
            content = message.content[0].content
            try:
                parsed = json.loads(content)
                formatted = json.dumps(parsed, indent=4)
            except:
                # Fallback to eval if it was a python-like repr, but safer to just use string
                try:
                    parsed = eval(content)
                    formatted = json.dumps(parsed, indent=4)
                except:
                    formatted = content

            return f"⚙️(id={message.content[0].call_id}) \n{_truncate_base64(formatted)}"
        except Exception as e:
            return f"⚙️(id={message.content[0].call_id}) \nError rendering result: {e}"

    elif isinstance(message, ToolCallSummaryMessage) and len(message.content) >= 1:
        return _truncate_base64("\n".join(message.content))
    elif isinstance(message, GroupChatTermination):
        return _truncate_base64(message.message.content)
    else:
        if len(message.content) == 0:
            return "(It seems that there is no any response from the agent)"
        else:
            return _truncate_base64(f"{message.content}")


SyncInputFunc = Callable[[str], str]
AsyncInputFunc = Callable[[str, Optional[CancellationToken]], Awaitable[str]]
InputFuncType = Union[SyncInputFunc, AsyncInputFunc]

T = TypeVar("T", bound=TaskResult | Response)


# Initialize colorama
init(autoreset=True)


def aprint(
    source, output: str, end: str = "\n", flush: bool = False
) -> Awaitable[None]:
    """
    Asynchronously print colored output based on the source.

    Args:
        source: The source identifier (e.g., "hypothesis", "search", etc.)
        output: The text to print
        end: The string appended after the output (default: newline)
        flush: Whether to force flush the output (default: False)

    Returns:
        Awaitable for the print operation
    """
    # Define color mappings for different agent roles
    color_map = {
        "user_proxy": Fore.CYAN,
        "hypothesis": Fore.YELLOW,
        "search": Fore.GREEN,
        "experiment": Fore.MAGENTA,
        "principle": Fore.BLUE,
    }

    # Determine the source identifier
    if isinstance(source, str):
        source_name = source.lower()
    else:
        # Try to get the class name if source is an object
        try:
            source_name = source.__class__.__name__.lower()
        except (AttributeError, TypeError):
            source_name = str(source).lower()

    # Select appropriate color based on source
    color = Fore.WHITE  # Default color
    for role, role_color in color_map.items():
        if role in source_name:
            color = role_color
            break

    # Apply color to the output
    colored_output = f"{color}{output}{Style.RESET_ALL}"

    # Run print asynchronously
    return asyncio.to_thread(print, colored_output, end=end, flush=flush)


async def Console(
    stream: AsyncGenerator[AgentEvent | ChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    user_input_manager: UserInputManager | None = None,
) -> T:
    """
    Consumes the message stream from :meth:`~autogen_agentchat.base.TaskRunner.run_stream`
    or :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream` and renders the messages to the console.
    Returns the last processed TaskResult or Response.

    Args:
        stream (AsyncGenerator[AgentEvent | ChatMessage | TaskResult, None] | AsyncGenerator[AgentEvent | ChatMessage | Response, None]): Message stream to render.
            This can be from :meth:`~autogen_agentchat.base.TaskRunner.run_stream` or :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream`.
        no_inline_images (bool, optional): If terminal is iTerm2 will render images inline. Use this to disable this behavior. Defaults to False.
        output_stats (bool, optional): (Experimental) If True, will output a summary of the messages and inline token usage info. Defaults to False.

    Returns:
        last_processed: A :class:`~autogen_agentchat.base.TaskResult` if the stream is from :meth:`~autogen_agentchat.base.TaskRunner.run_stream`
            or a :class:`~autogen_agentchat.base.Response` if the stream is from :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream`.
    """
    render_image_iterm = (
        _is_running_in_iterm() and _is_output_a_tty() and not no_inline_images
    )
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    last_processed: Optional[T] = None

    streaming_chunks: List[str] = []

    async for message in stream:
        if isinstance(message, TaskResult):
            duration = time.time() - start_time
            if output_stats:
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of messages: {len(message.messages)}\n"
                    f"Finish reason: {message.stop_reason}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(message, output, end="", flush=True)

            # mypy ignore
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = time.time() - start_time

            # Print final response.
            output = f"{'-' * 10} {message.chat_message.source} {'-' * 10}\n{_message_to_str(message.chat_message, render_image_iterm=render_image_iterm)}\n"

            if message.chat_message.models_usage:
                if output_stats:
                    output += f"[Prompt tokens: {message.chat_message.models_usage.prompt_tokens}, Completion tokens: {message.chat_message.models_usage.completion_tokens}]\n"
                total_usage.completion_tokens += (
                    message.chat_message.models_usage.completion_tokens
                )
                total_usage.prompt_tokens += (
                    message.chat_message.models_usage.prompt_tokens
                )
            await aprint(message.chat_message.source, output, end="", flush=True)

            # Print summary.
            if output_stats:
                if message.inner_messages is not None:
                    num_inner_messages = len(message.inner_messages)
                else:
                    num_inner_messages = 0
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of inner messages: {num_inner_messages}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(message.chat_message.source, output, end="", flush=True)

            # mypy ignore
            last_processed = message  # type: ignore

        # We don't want to print UserInputRequestedEvent messages, we just use them to signal the user input event.
        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)

        else:
            # Cast required for mypy to be happy
            if not streaming_chunks:
                # Print message sender.
                await aprint(
                    message.source,
                    f"{'-' * 10} {message.__class__.__name__} ({message.source}) {'-' * 10}",
                    end="\n",
                    flush=True,
                )
            if isinstance(message, ModelClientStreamingChunkEvent):
                # Truncate chunks if they contain base64 (less common but possible)
                await aprint(
                    source=message.source,
                    output=_truncate_base64(message.to_text()),
                    end="",
                )
                streaming_chunks.append(message.content)
            else:
                if streaming_chunks:
                    streaming_chunks.clear()
                    # Chunked messages are already printed, so we just print a newline.
                    await aprint(message.source, "", end="\n", flush=True)
                elif isinstance(message, MultiModalMessage):
                    # MultiModalMessage to_text already handles iterm, but we wrap in truncation for safety
                    await aprint(
                        message.source,
                        _truncate_base64(message.to_text(iterm=render_image_iterm)),
                        end="\n",
                        flush=True,
                    )
                elif isinstance(message, ToolCallExecutionEvent):
                    # Specialized handling for ToolCallExecutionEvent to ensure truncation
                    for result in message.content:
                        truncated_content = _truncate_base64(str(result.content))
                        output = f"⚙️(id={result.call_id}) \n{truncated_content}"
                        await aprint(message.source, output, end="\n", flush=True)
                elif isinstance(message, ToolCallRequestEvent):
                    for call in message.content:
                        output = f"🔧(id={call.id}) \n{call.name}({call.arguments})"
                        await aprint(message.source, output, end="\n", flush=True)
                else:
                    await aprint(
                        message.source,
                        _truncate_base64(message.to_text()),
                        end="\n",
                        flush=True,
                    )
                if message.models_usage:
                    if output_stats:
                        await aprint(
                            message.source,
                            f"[Prompt tokens: {message.models_usage.prompt_tokens}, Completion tokens: {message.models_usage.completion_tokens}]",
                            end="\n",
                            flush=True,
                        )
                    total_usage.completion_tokens += (
                        message.models_usage.completion_tokens
                    )
                    total_usage.prompt_tokens += message.models_usage.prompt_tokens

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")

    return last_processed
