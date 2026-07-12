"""Compatibility facade for class-based runtime-environment tools."""

from openai4s.tools.env_create import EnvCreateTool
from openai4s.tools.env_list import EnvListTool
from openai4s.tools.env_use import EnvUseTool
from openai4s.tools.registry import get_tool

env_list = get_tool("env_list")
env_use = get_tool("env_use")
env_create = get_tool("env_create")

__all__ = [
    "EnvListTool",
    "EnvUseTool",
    "EnvCreateTool",
    "env_list",
    "env_use",
    "env_create",
]
