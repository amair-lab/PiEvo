import json
import time
import ast
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Counter, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class NoteTaker:
    """A class to record chat history from different agents in a structured format.

    The class ignores ModelClientStreamingChunkEvent messages and focuses on
    complete messages. Creates a standardized format with consistent fields
    that can be easily converted to a dataframe for analysis.
    """

    def __init__(self, output_file: str = "chat_history.json"):
        """Initialize the NoteTaker.

        Args:
            output_file (str): Path to the JSON file to save the chat history.
        """
        self.output_file = output_file
        self.start_time = time.time()

        # Standardized message records
        self.messages = []

        # Speaker statistics
        self.speaker_counter = Counter()
        self.unique_speakers = set()

        # Tool usage tracking
        self.tool_calls = []
        self.experiment_results = []

    def record_message(self, message: Any) -> None:
        """Record a message or event, skipping streaming token events.

        Args:
            message: The message or event to record.
        """
        # Debug: Log what we're receiving
        message_type = type(message).__name__
        message_source = getattr(message, "source", "no_source")
        logger.debug(f"📝 NoteTaker received: {message_type} from {message_source}")

        # Skip ModelClientStreamingChunkEvent messages entirely
        if message_type == "ModelClientStreamingChunkEvent":
            logger.debug(f"⏭️ Skipping streaming chunk from {message_source}")
            return

        # Process the message
        try:
            message_info = self._extract_message_info(message)
            if message_info:
                self.messages.append(message_info)
                logger.debug(
                    f"✅ Recorded {message_type} from {message_source} (total: {len(self.messages)})"
                )

                # Update speaker statistics if source is available
                if "source" in message_info and message_info["source"]:
                    speaker = message_info["source"]
                    self.speaker_counter[speaker] += 1
                    self.unique_speakers.add(speaker)
            else:
                logger.warning(
                    f"❌ No message info extracted for {message_type} from {message_source}"
                )

        except Exception as e:
            logger.error(f"⚠️ Error recording message {message_type}: {e}")
            # Create a safe fallback record
            fallback_record = {
                "timestamp": time.time() - self.start_time,
                "source": getattr(message, "source", "unknown"),
                "type": message_type,
                "content": f"Error recording message: {str(e)}",
                "tool_call": None,
                "tool_response": None,
                "error": str(e),
            }
            self.messages.append(fallback_record)
            logger.debug(
                f"✅ Added fallback record for {message_type} (total: {len(self.messages)})"
            )

    def _extract_message_info(self, message: Any) -> Optional[Dict[str, Any]]:
        """Extract relevant information from a message or event in a standardized format.

        Args:
            message: The message or event to extract information from.

        Returns:
            Dict or None: A dictionary with standardized fields.
        """
        message_type = type(message).__name__

        # Create a standardized message record with all fields initialized
        message_info = {
            "timestamp": time.time() - self.start_time,
            "source": getattr(message, "source", "unknown"),
            "type": message_type,
            "content": None,
            "tool_call": None,
            "tool_response": None,
        }

        # Extract content if available
        if hasattr(message, "content"):
            message_info["content"] = self._make_serializable(message.content)
            logger.debug(
                f"🔍 Extracted content from {message_type}: {str(message_info['content'])[:100]}..."
            )

        # Handle different message types based on class name
        if message_type == "ToolCallRequestEvent":
            logger.debug(
                f"🔧 Processing ToolCallRequestEvent from {message_info['source']}"
            )
            # Extract tool calls
            if hasattr(message, "content") and isinstance(message.content, list):
                tool_calls = []
                for call in message.content:
                    call_info = {}

                    # Extract name, arguments and id
                    if hasattr(call, "name"):
                        call_info["name"] = call.name

                    if hasattr(call, "arguments"):
                        call_info["arguments"] = self._safe_parse_arguments(
                            call.arguments
                        )

                    if hasattr(call, "id"):
                        call_info["id"] = call.id

                    tool_calls.append(call_info)

                message_info["tool_call"] = tool_calls
                logger.debug(f"🔧 Extracted {len(tool_calls)} tool calls")

        elif message_type == "ToolCallExecutionEvent":
            logger.debug(
                f"⚙️ Processing ToolCallExecutionEvent from {message_info['source']}"
            )
            # Extract tool responses
            tool_responses = []
            if hasattr(message, "content") and isinstance(message.content, list):
                for result in message.content:
                    response_info = {}

                    # Extract call_id and error status
                    if hasattr(result, "call_id"):
                        response_info["call_id"] = result.call_id

                    if hasattr(result, "is_error"):
                        response_info["is_error"] = result.is_error

                    # Extract content safely
                    if hasattr(result, "content"):
                        response_info["content"] = self._safe_parse_tool_content(
                            result.content
                        )

                    tool_responses.append(response_info)

            # Add tool responses to the message record
            message_info["tool_response"] = tool_responses
            logger.debug(f"⚙️ Extracted {len(tool_responses)} tool responses")

            # Also add to specialized tool_calls list
            self.tool_calls.append(
                {
                    "timestamp": message_info["timestamp"],
                    "source": message_info["source"],
                    "responses": tool_responses,
                }
            )

        elif message_type == "ToolCallSummaryMessage":
            logger.debug(
                f"📊 Processing ToolCallSummaryMessage from {message_info['source']}"
            )
            # Special handling for experiment agent's output
            if message_info["source"] == "experiment":
                try:
                    parsed_content = self._safe_parse_tool_content(
                        message_info["content"]
                    )

                    # Add to specific experiment results list
                    experiment_result = {
                        "timestamp": message_info["timestamp"],
                        "data": parsed_content,
                    }
                    self.experiment_results.append(experiment_result)
                    logger.debug(f"🧪 Added experiment result to specialized list")
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse experiment content: {e}")

        elif message_type == "TextMessage":
            logger.debug(f"💬 Processing TextMessage from {message_info['source']}")
            # Standard text message - content already extracted

        elif message_type == "TaskResult":
            logger.debug(f"🏁 Processing TaskResult - final result")
            # Task result - extract stop reason if available
            if hasattr(message, "stop_reason"):
                message_info["stop_reason"] = message.stop_reason

        else:
            logger.debug(f"❓ Processing unknown message type: {message_type}")

        # Always return the message_info - don't return None unless absolutely necessary
        logger.debug(
            f"✅ Message info extracted for {message_type}: source={message_info['source']}, has_content={message_info['content'] is not None}"
        )
        return message_info

    def _safe_parse_arguments(self, arguments: Any) -> Any:
        """Safely parse tool call arguments."""
        try:
            if isinstance(arguments, str):
                # Try to parse as JSON first
                try:
                    return json.loads(arguments)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    try:
                        return ast.literal_eval(arguments)
                    except (ValueError, SyntaxError):
                        # If all parsing fails, return as string
                        return arguments
            return self._make_serializable(arguments)
        except Exception as e:
            logger.warning(f"⚠️ Error parsing arguments: {e}")
            return str(arguments)

    def _safe_parse_tool_content(self, content: Any) -> Any:
        """Safely parse tool call content without unsafe eval()."""
        try:
            if isinstance(content, str):
                content_str = content.strip()

                # Try JSON parsing first
                if content_str.startswith("{") and content_str.endswith("}"):
                    try:
                        return json.loads(content_str)
                    except json.JSONDecodeError:
                        pass

                # Try ast.literal_eval for safe evaluation
                try:
                    return ast.literal_eval(content_str)
                except (ValueError, SyntaxError):
                    pass

                # If content contains numpy references, try to handle them
                if "np." in content_str or "numpy." in content_str:
                    try:
                        # Create a safe namespace with numpy
                        safe_namespace = {"np": np, "numpy": np, "__builtins__": {}}
                        # Use eval with restricted namespace
                        return eval(content_str, safe_namespace)
                    except Exception as eval_error:
                        logger.warning(
                            f"⚠️ Could not evaluate numpy expression: {eval_error}"
                        )
                        return content_str

                # Return as string if all parsing fails
                return content_str

            return self._make_serializable(content)

        except Exception as e:
            logger.warning(f"⚠️ Error parsing tool content: {e}")
            return str(content) if content is not None else "None"

    def _make_serializable(self, obj: Any) -> Any:
        """Convert a potentially non-serializable object to a JSON-serializable representation.

        Args:
            obj: The object to make serializable

        Returns:
            A JSON-serializable version of the object
        """
        if obj is None:
            return None

        # Handle basic serializable types
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # Handle numpy types
        if hasattr(obj, "dtype"):  # numpy array or scalar
            try:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                else:
                    return float(obj)
            except Exception:
                return str(obj)

        # Handle lists and tuples recursively
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]

        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}

        # Handle objects with __dict__ attribute
        if hasattr(obj, "__dict__"):
            # Extract only the most important attributes to avoid excessive nesting
            if hasattr(obj, "type") and hasattr(obj, "content"):
                return {
                    "type": obj.type,
                    "content": self._make_serializable(obj.content),
                }
            return self._make_serializable(obj.__dict__)

        # For other types, convert to string representation
        try:
            return str(obj)
        except Exception:
            return f"<Non-serializable object of type {type(obj).__name__}>"

    def get_speaker_stats(self) -> Dict[str, Any]:
        """Get statistics about speakers in the conversation.

        Returns:
            Dict: Statistics about speaker participation
        """
        total_messages = sum(self.speaker_counter.values())

        stats = {
            "total_messages": total_messages,
            "unique_speakers": len(self.unique_speakers),
            "speaker_counts": dict(self.speaker_counter),
            "speaker_percentages": {
                speaker: (count / total_messages * 100) if total_messages > 0 else 0
                for speaker, count in self.speaker_counter.items()
            },
        }

        return stats

    def save(self) -> None:
        """Save the recorded messages and statistics to a JSON file."""
        logger.info(f"💾 Saving NoteTaker data...")
        logger.info(f"📊 Total messages recorded: {len(self.messages)}")
        logger.info(f"🔧 Total tool calls: {len(self.tool_calls)}")
        logger.info(f"🧪 Total experiment results: {len(self.experiment_results)}")

        try:
            # Generate speaker statistics
            speaker_stats = self.get_speaker_stats()
            logger.debug(f"👥 Speaker stats: {speaker_stats}")

            # Prepare the output data
            output = {
                "metadata": {
                    "recording_duration": time.time() - self.start_time,
                    "total_messages": len(self.messages),
                    "timestamp": time.time(),
                    "unique_speakers": len(self.unique_speakers),
                    "speaker_list": list(self.unique_speakers),
                },
                "speaker_statistics": speaker_stats,
                "messages": self.messages,
                "tool_calls": self.tool_calls,
                "experiment_results": self.experiment_results,
            }

            # Save main file
            with open(self.output_file, "w") as f:
                json.dump(output, f, indent=4)
            logger.info(f"✅ Main file saved: {self.output_file}")

            # Also save a simplified version that's more amenable to dataframe conversion
            df_ready = []
            for msg in self.messages:
                # Create a flattened record with consistent columns
                record = {
                    "timestamp": msg.get("timestamp", 0),
                    "source": msg.get("source", "unknown"),
                    "type": msg.get("type", "unknown"),
                    "content": (
                        str(msg.get("content", ""))
                        if msg.get("content") is not None
                        else ""
                    ),
                    # Convert complex objects to simple indicators for dataframe use
                    "has_tool_call": msg.get("tool_call") is not None,
                    "has_tool_response": msg.get("tool_response") is not None,
                }
                df_ready.append(record)

            dataframe_file = self.output_file.replace(".json", "_dataframe.json")
            with open(dataframe_file, "w") as f:
                json.dump(df_ready, f, indent=4)
            logger.info(f"✅ Dataframe file saved: {dataframe_file}")

            # Save a debug summary
            debug_summary = {
                "debug_info": {
                    "total_messages_processed": len(self.messages),
                    "message_types": {},
                    "speakers": list(self.unique_speakers),
                    "save_timestamp": time.time(),
                }
            }

            # Count message types
            for msg in self.messages:
                msg_type = msg.get("type", "unknown")
                if msg_type not in debug_summary["debug_info"]["message_types"]:
                    debug_summary["debug_info"]["message_types"][msg_type] = 0
                debug_summary["debug_info"]["message_types"][msg_type] += 1

            debug_file = self.output_file.replace(".json", "_debug.json")
            with open(debug_file, "w") as f:
                json.dump(debug_summary, f, indent=4)
            logger.debug(f"🐛 Debug file saved: {debug_file}")

            logger.info(f"💾 All files saved successfully!")

        except Exception as e:
            logger.error(f"⚠️ Error during save: {e}")
            logger.info(
                f"📊 Data state: messages={len(self.messages)}, tool_calls={len(self.tool_calls)}"
            )

            # Attempt a safe save with string conversion
            try:
                safe_output = {
                    "metadata": {
                        "recording_duration": time.time() - self.start_time,
                        "total_messages": len(self.messages),
                        "timestamp": time.time(),
                        "save_error": str(e),
                        "safe_mode": True,
                    },
                    "messages_summary": {
                        "count": len(self.messages),
                        "last_10_messages": [
                            {
                                "type": msg.get("type", "unknown"),
                                "source": msg.get("source", "unknown"),
                                "content_preview": str(msg.get("content", ""))[:100],
                            }
                            for msg in self.messages[-10:]
                        ],
                    },
                    "error_details": str(e),
                }

                safe_file = self.output_file.replace(".json", "_safe.json")
                with open(safe_file, "w") as f:
                    json.dump(safe_output, f, indent=4)

                logger.info(f"💾 Safe save completed: {safe_file}")
            except Exception as safe_error:
                logger.error(f"❌ Even safe save failed: {safe_error}")

                # Last resort: write plain text
                try:
                    text_file = self.output_file.replace(".json", "_emergency.txt")
                    with open(text_file, "w") as f:
                        f.write(f"Emergency NoteTaker Save\n")
                        f.write(f"Time: {time.time()}\n")
                        f.write(f"Messages recorded: {len(self.messages)}\n")
                        f.write(f"Tool calls: {len(self.tool_calls)}\n")
                        f.write(f"Speakers: {list(self.unique_speakers)}\n")
                        f.write(f"Error: {e}\n")
                        f.write(f"Safe error: {safe_error}\n")
                    logger.info(f"📄 Emergency text file saved: {text_file}")
                except Exception as text_error:
                    logger.critical(f"💥 Total save failure: {text_error}")
