"""Read-only Store metadata and lineage control tools.

The Store remains the SQL/query safety boundary. These classes expose its
existing HostDispatcher adapters to the native JSON control plane without
duplicating persistence behavior or granting write-capable SQL.
"""

from __future__ import annotations

from typing import Any

from openai4s.tools.base import Tool
from openai4s.tools.contexts import ControlToolContext
from openai4s.tools.taxonomy import resource_key


class QuerySchemaTool(Tool):
    """Inspect the allowlisted read-only Store schema."""

    name = "query_schema"
    host_method = "query_schema"
    description = "List readable Store tables and their columns."
    parameters = {"properties": {}, "required": []}
    requires_approval = False
    resource_key_prefix = "database"
    resource_target_default = "schema"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        del arguments
        return runtime.invoke(self.host_method)


class ReadOnlyQueryTool(Tool):
    """Run the Store's guarded SELECT/CTE query path."""

    name = "query"
    host_method = "query"
    description = (
        "Run a bounded read-only SQL SELECT or CTE over allowlisted Store tables."
    )
    parameters = {
        "properties": {
            "sql": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100000,
                "description": "A read-only SELECT or CTE statement.",
            },
            "params": {
                "type": "array",
                "items": {},
                "maxItems": 1000,
                "description": "Positional SQLite parameters.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "Maximum rows returned.",
            },
            "df": {
                "type": "boolean",
                "description": "Return columns plus row arrays instead of objects.",
            },
        },
        "required": ["sql"],
    }
    requires_approval = False
    resource_key_prefix = "database"
    resource_target_default = "query"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> Any:
        return runtime.invoke(self.host_method, dict(arguments))


class FramesTool(Tool):
    """Browse, search, or inspect persisted research frames."""

    name = "frames"
    host_method = "frames"
    description = "Browse, search, or inspect persisted agent/session frames."
    parameters = {
        "properties": {
            "frame_id": {"type": "string", "minLength": 1, "maxLength": 256},
            "pattern": {"type": "string", "minLength": 1, "maxLength": 1000},
            "project_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "status": {
                "type": "string",
                "enum": [
                    "processing",
                    "done",
                    "failed",
                    "awaiting_user_response",
                ],
            },
            "roots_only": {"type": "boolean"},
            "page": {"type": "integer", "minimum": 0, "maximum": 100000},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": [],
    }
    requires_approval = False
    resource_key_prefix = "frame"
    resource_target_default = "catalog"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        spec = arguments if isinstance(arguments, dict) else {}
        target = spec.get("frame_id") or spec.get("project_id") or "catalog"
        return (resource_key("frame", target),)

    def execute(self, runtime: ControlToolContext, arguments: dict) -> Any:
        return runtime.invoke(self.host_method, dict(arguments))


class LineageGetTool(Tool):
    """Read the immediate provenance record for one Artifact version."""

    name = "lineage_get"
    host_method = "lineage_get"
    description = "Get provenance and direct inputs for one artifact version."
    parameters = {
        "properties": {
            "version_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            }
        },
        "required": ["version_id"],
    }
    requires_approval = False
    resource_key_prefix = "lineage"
    resource_target_key = "version_id"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        version_id = (
            arguments
            if isinstance(arguments, str)
            else (arguments or {}).get("version_id")
        )
        return (resource_key("lineage", version_id),)

    def execute(
        self,
        runtime: ControlToolContext,
        arguments: dict | str,
    ) -> dict:
        version_id = (
            arguments if isinstance(arguments, str) else arguments["version_id"]
        )
        return runtime.invoke(self.host_method, version_id)


class LineageGraphTool(Tool):
    """Traverse a bounded Artifact-version lineage graph."""

    name = "lineage_graph"
    host_method = "lineage_graph"
    description = "Traverse upstream or downstream artifact-version lineage."
    parameters = {
        "properties": {
            "version_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "direction": {"type": "string", "enum": ["up", "down"]},
            "max_depth": {"type": "integer", "minimum": 0, "maximum": 100},
            "max_nodes": {"type": "integer", "minimum": 1, "maximum": 10000},
        },
        "required": ["version_id"],
    }
    requires_approval = False
    resource_key_prefix = "lineage"
    resource_target_key = "version_id"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, dict(arguments))


__all__ = [
    "FramesTool",
    "LineageGetTool",
    "LineageGraphTool",
    "QuerySchemaTool",
    "ReadOnlyQueryTool",
]
