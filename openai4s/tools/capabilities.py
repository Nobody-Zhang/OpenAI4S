"""Active discovery for progressively disclosed native tool groups."""

from __future__ import annotations

from openai4s.tools.base import Tool
from openai4s.tools.contexts import ControlToolContext
from openai4s.tools.taxonomy import RUNTIME_MUTATION


class SearchCapabilitiesTool(Tool):
    """Search and monotonically activate matching SessionToolCatalog groups."""

    name = "search_capabilities"
    host_method = "search_capabilities"
    description = (
        "Search hidden control-tool capability groups and activate matching tools "
        "for this session."
    )
    parameters = {
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["query"],
    }
    read_only = False
    requires_approval = False
    side_effect_class = RUNTIME_MUTATION
    resource_key_prefix = "capability"
    resource_target_default = "catalog"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, {"query": arguments["query"]})


__all__ = ["SearchCapabilitiesTool"]
