"""Compatibility facade for class-based web control tools."""

from openai4s.tools.registry import get_tool
from openai4s.tools.web_fetch import WebFetchTool
from openai4s.tools.web_search import WebSearchTool

web_search = get_tool("web_search")
web_fetch = get_tool("web_fetch")

__all__ = [
    "WebSearchTool",
    "WebFetchTool",
    "web_search",
    "web_fetch",
]
