"""
Automatic tool loading system using decorators.
"""

import importlib
import pkgutil
from typing import Dict, Any, Callable, List

# Global registry to store all registered tools
_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, description: str):
    """
    Decorator to register a function as a tool.

    Args:
        name: The name of the tool (used in config files)
        description: Description of what the tool does

    Usage:
        @register_tool("tool_name", "Description of the tool")
        def my_tool_function(param1: str, param2: int) -> str:
            return "result"
    """

    def decorator(func: Callable) -> Callable:
        # Store the tool metadata in the global registry
        _TOOL_REGISTRY[name] = {
            "function": func,
            "description": description,
            "name": name,
        }

        # Add metadata to the function itself for introspection
        func._tool_name = name
        func._tool_description = description

        return func

    return decorator


def get_all_tools() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered tools.

    Returns:
        Dictionary mapping tool names to their metadata
    """
    return _TOOL_REGISTRY.copy()


def get_tool(name: str) -> Dict[str, Any]:
    """
    Get a specific tool by name.

    Args:
        name: The tool name

    Returns:
        Tool metadata dictionary or None if not found
    """
    return _TOOL_REGISTRY.get(name)


def discover_and_load_tools():
    """
    Automatically discover and import all tool modules to trigger registration.
    This should be called before accessing tools.
    """
    import pievo.tools as tools

    # Get the tools package path
    package_path = tools.__path__

    # Iterate through all modules in the tools package
    for finder, module_name, ispkg in pkgutil.iter_modules(package_path, "pievo.tools."):
        if module_name != "pievo.tools.__init__":  # Skip self
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"Warning: Could not import tool module {module_name}: {e}")


def create_function_tools_dict():
    """
    Create a dictionary of FunctionTool objects for all registered tools.
    This is compatible with the existing inference.py structure.

    Returns:
        Dictionary mapping tool names to FunctionTool objects
    """
    from autogen_core.tools import FunctionTool

    # First, ensure all tools are discovered and loaded
    discover_and_load_tools()

    # Create FunctionTool objects for all registered tools
    function_tools = {}
    for tool_name, tool_info in _TOOL_REGISTRY.items():
        function_tools[tool_name] = FunctionTool(
            tool_info["function"], description=tool_info["description"]
        )

    return function_tools


# Auto-discover tools when this module is imported
discover_and_load_tools()
