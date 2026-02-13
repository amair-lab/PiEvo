#!/usr/bin/env python3
import asyncio
import os
import sys
import tempfile
import time

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

from autogen_core.model_context import BufferedChatCompletionContext

from pievo.agents import UserProxy, HypothesisAgent, ExperimentAgent, PrincipleAgent

from pievo.group.manage import SubmissionBasedGroupChat

from pievo import tools
from pievo.utils.config import (
    load_config,
    init_results,
)

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console

from autogen_core.models import (
    ModelFamily,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache

from pievo.utils.console import Console
from pievo.group.evolveflow import PiEvo

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(message)s"
)

evolve_logger = logging.getLogger("pievo.group.evolveflow")
evolve_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
evolve_logger.addHandler(handler)
evolve_logger.propagate = False

class PiFlow:
    def __init__(
        self,
        args,
        task_cfg_path,
        model_cfg_path,
        save_dir: Optional[str] = "./",
    ) -> None:
        self.args = args

        self.model_config = load_config(model_cfg_path)
        self.task_config = load_config(task_cfg_path)

        # Set cache directory based on enable_cache flag
        self.cache_dir = args.output_dir
        self.save_dir = save_dir
        self.off_pievo = args.off_pievo

        self.agent_base_url = args.agent_base_url_force
        self.agent_model_name = args.agent_model_name_force
        self.agent_temperature = args.agent_temperature_force
        self.agent_max_tokens = args.agent_max_tokens_force
        self.agent_api_key = args.agent_api_key_force

        init_results(
            self.save_dir,
            model_cfg_path,
            self.model_config,
            task_cfg_path,
            self.task_config,
        )

        self.agents: Dict[str, Union[AssistantAgent, UserProxyAgent]] = {}

        # Initialize cache storage only when cache is enabled
        if self.cache_dir:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            # Create a shared cache storage instance for all clients
            util_cache_dir = os.path.join(self.cache_dir, "util_client")
            os.makedirs(util_cache_dir, exist_ok=True)
            self.cache_storage = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](
                Cache(directory=util_cache_dir)
            )
        else:
            self.cache_storage = None

        for key in self.task_config.get("environment").keys():
            os.environ[key] = self.task_config.get("environment")[key]

        # Automatically load all registered tools
        self.available_tools = tools.create_function_tools_dict()

        self._set_util_client(cache_dir=self.cache_dir)


        # If off PiEvo, this term will be None. 
        self.pievo = PiEvo(
            task=self.task_config.get("task"),
            output_file=os.path.join(save_dir, f"submissions.json"),
            client=self.util_client,
            output_dir=args.output_dir,
            sigma=args.sigma,
            anomaly_threshold=args.anomaly_threshold,
            warm_up_rounds=args.warm_up_rounds,
            new_principle_prior_mass=args.new_principle_prior_mass,
            off_pievo=self.off_pievo,
        )

        # Agents will be created asynchronously using the factory method
        # so team will be created in the async setup method

    def _create_team(self):
        """Create the team after agents have been initialized."""
        agent_rotation_order = ["principle", "hypothesis", "experiment"]
        
        # Create submission patterns for each agent type
        submission_patterns = {
            "hypothesis": r"HYPOTHESIS_SUBMISSION\s*(.+)",
            "experiment": r"EXPERIMENT_SUBMISSION\s*(.+)",
            "principle": r"PRINCIPLE_SUBMISSION\s*(.+)",
        }
        
        self.team = SubmissionBasedGroupChat(
            participants=[agent for agent in self.agents.values()],
            model_client=self.util_client,  # Required for SelectorGroupChat
            max_turns=self.args.max_turn,
            termination_condition=TextMentionTermination("TERMINATE"),
            note_taker_output_file=os.path.join(self.save_dir, "running_notes.json"),
            agent_rotation_order=agent_rotation_order,
            submission_patterns=submission_patterns,
            allow_repeated_speaker=True,  # Enable consecutive speaking
            selector_prompt="Custom submission-based speaker selection for multi-agent workflow",
        )

        print(f"🎯 Configured submission-based group chat:")
        print(f"📋 Agent rotation: {agent_rotation_order}")
        print(f"🔍 Submission patterns: {list(submission_patterns.keys())}")

    @property
    def task(self):
        return self.task_config.get("task")

    def _set_util_client(self, cache_dir: Optional[str] = None) -> None:
        """Set up utility client with optional caching support."""
        # Util client serve as a processing tool for analysis with language.
        openai_model_client = OpenAIChatCompletionClient(
            api_key=os.getenv("UTIL_LLM_CONFIG_API_KEY"),
            base_url=os.getenv("UTIL_LLM_CONFIG_BASE_URL"),
            model=os.getenv("UTIL_LLM_CONFIG_NAME"),
            temperature=float(os.getenv("UTIL_LLM_CONFIG_TEMPERATURE")),
            max_tokens=int(os.getenv("UTIL_LLM_CONFIG_MAX_TOKENS")),
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.GPT_4,
                "structured_output": False,
            },
        )

        if cache_dir is not None:
            # Create separate cache directory for util client
            util_cache_dir = os.path.join(cache_dir, "util_client")
            os.makedirs(util_cache_dir, exist_ok=True)
            util_cache_storage = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](
                Cache(directory=util_cache_dir)
            )
            self.util_client: OpenAIChatCompletionClient | ChatCompletionCache = (
                ChatCompletionCache(openai_model_client, util_cache_storage)
            )
        else:
            self.util_client: OpenAIChatCompletionClient | ChatCompletionCache = (
                openai_model_client
            )

    def create_client(
        self,
        llm_config: Dict[str, Any],
        cache_dir: Optional[str] = None,
        agent_name: str = "default",
    ) -> OpenAIChatCompletionClient | ChatCompletionCache:
        """Create an OpenAIChatCompletionClient instance based on LLM configuration with separate cache storage per agent."""

        api_key = llm_config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("API key is required for OpenAIChatCompletionClient")

        openai_client: OpenAIChatCompletionClient = OpenAIChatCompletionClient(
            api_key=api_key if not self.agent_api_key else self.agent_api_key,
            base_url=llm_config.get("base_url", "https://api.openai.com/v1") if not self.agent_base_url else self.agent_base_url,
            model=llm_config.get("model_name", "gpt-4o") if not self.agent_model_name else self.agent_model_name,
            temperature=llm_config.get("temperature", 0.7) if not self.agent_temperature else self.agent_temperature,
            max_tokens=llm_config.get("max_tokens", 2048) if not self.agent_max_tokens else self.agent_max_tokens,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": llm_config.get("json_output", False),
                "family": (
                    ModelFamily.R1
                    if llm_config.get("is_reasoning", False)
                    else ModelFamily.GEMINI_2_5_PRO
                ),
                "structured_output": False,
            },
        )

        if cache_dir is not None:
            # Create separate cache directory for each agent to avoid conflicts
            agent_cache_dir = os.path.join(cache_dir, f"agent_{agent_name}")
            os.makedirs(agent_cache_dir, exist_ok=True)
            agent_cache_storage = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](
                Cache(directory=agent_cache_dir)
            )
            return ChatCompletionCache(openai_client, agent_cache_storage)
        else:
            return openai_client


    async def _create_agents(self) -> None:
        """Create all agents based on the configuration."""
        agent_classes = {
            "user_proxy": UserProxy,
            "hypothesis": HypothesisAgent,
            "experiment": ExperimentAgent,
            "principle": PrincipleAgent,
        }
        # Iterate over agent classes to instantiate them
        for agent_name, agent_class in agent_classes.items():
            _is_code_mode = False

            if "user_proxy" == agent_name:
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})

                if not agent_config or not agent_config.get("enabled"):
                    continue

                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    description="Human user",
                    input_func=None,
                )

            # Coding Agent with DockerCodeTool support.
            elif "experiment" == agent_name and _is_code_mode:
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})
                docker_executor = DockerCommandLineCodeExecutor(
                    image="jupyter/datascience-notebook:latest",
                    auto_remove=True,
                    stop_container=True,
                )
                await docker_executor.start()

                code_tool = PythonCodeExecutionTool(docker_executor)
                llm_config = agent_config.get("api_config", {})
                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    system_message=agent_config.get("system_prompt", None),
                    description=agent_config.get("description", ""),
                    tools=[code_tool, ],
                    model_context=BufferedChatCompletionContext(
                        buffer_size=agent_config.get("message_buffer_size", 10)
                    ),
                    max_tool_iterations=5,
                    reflect_on_tool_use=True,
                    strategy=self.pievo,
                    model_client=self.create_client(
                        llm_config=llm_config,
                        cache_dir=self.cache_dir,
                        agent_name=agent_name,
                    ),
                )
            elif (
                self.model_config.get("agents", {})
                .get(agent_name, {})
                .get("enabled", False)
            ):
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})
                llm_config = agent_config.get("api_config", {})

                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    description=agent_config.get("description", ""),
                    system_message=agent_config.get("system_prompt", None),
                    model_client=self.create_client(
                        llm_config=llm_config,
                        cache_dir=self.cache_dir,
                        agent_name=agent_name,
                    ),
                    model_client_stream=agent_config.get("streaming", False),
                    tools=[
                        self.available_tools[_] for _ in agent_config.get("tools", [])
                    ],
                    model_context=BufferedChatCompletionContext(
                        buffer_size=agent_config.get("message_buffer_size", 10)
                    ),
                    strategy=self.pievo,
                )


    @classmethod
    async def create(
        cls,
        args,
        task_cfg_path,
        model_cfg_path,
        save_dir: str = "./",
    ) -> "PiFlow":
        self = cls(args, task_cfg_path, model_cfg_path, save_dir)
        await self._create_agents()
        self._create_team()
        return self


async def run_pievo():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser()

    # PIEVO
    parser.add_argument("--sigma", type=float,                      default=0.01, help="range: [0.01, 0.3, 0.6, 0.9]")
    parser.add_argument("--anomaly_threshold", type=float,          default=0.85, help="range: [0.15, 0.5, 0.85]")
    parser.add_argument("--warm_up_rounds", type=int,               default=10,   help="range: [5, 10, 15]")
    parser.add_argument("--new_principle_prior_mass", type=float,   default=1e-3, help="Fixed.")

    # INSTRUCTIONS
    parser.add_argument("--high_confidence_threshold", type=float,  default=0.9,  help="range: [0.6, 0.7, 0.8, 0.9]")
    parser.add_argument("--anomaly_cnt_threshold", type=int,        default=3,    help="range: [3, 5, 7]")

    # SYSTEM
    parser.add_argument("--max_turn", type=int,             required=True)
    parser.add_argument("--task_config", type=str,          required=True)
    parser.add_argument("--model_config", type=str,         required=True)
    parser.add_argument("--output_dir", type=str,           required=True)

    # OVERWITE AGENT_CONFIG
    parser.add_argument("--agent_api_key_force", type=str,          default=None)
    parser.add_argument("--agent_base_url_force", type=str,         default=None)
    parser.add_argument("--agent_model_name_force", type=str,       default=None)
    parser.add_argument("--agent_temperature_force", type=float,    default=None)
    parser.add_argument("--agent_max_tokens_force", type=int,       default=None)

    # ABLATION SETTINGS
    parser.add_argument("--off_pievo", action="store_true")


    args = parser.parse_args()

    os.environ["HIGH_CONFIDENCE_THRESHOLD"] = str(args.high_confidence_threshold)
    os.environ["NUM_EXPLORATION_CASES"] = str(args.warm_up_rounds)
    os.environ["ANOMALY_COUNTS_THRESHOLD"] = str(args.anomaly_cnt_threshold)

    os.environ["WORKING_RESULT_DIR"] = os.path.abspath(args.output_dir)

    piflow = await PiFlow.create(
        args,
        task_cfg_path=args.task_config,
        model_cfg_path=args.model_config,
        save_dir=args.output_dir,
    )

    stream = piflow.team.run_stream(task=piflow.task)
    await Console(stream, output_stats=True)


def main():
    asyncio.run(run_pievo())


if __name__ == "__main__":
    main()
