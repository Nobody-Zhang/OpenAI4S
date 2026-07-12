"""Artifact metadata control tools; scientific file production stays in code."""

from __future__ import annotations

from typing import Any

from openai4s.tools.base import Tool
from openai4s.tools.contexts import ControlToolContext
from openai4s.tools.taxonomy import resource_key


class ListArtifactsTool(Tool):
    """Search versioned Artifact metadata without reading arbitrary SQL."""

    name = "list_artifacts"
    host_method = "artifacts"
    description = "List or search versioned artifacts and their metadata."
    parameters = {
        "properties": {
            "search": {"type": "string", "minLength": 1},
            "artifact_id": {"type": "string", "minLength": 1},
            "root_frame_id": {"type": "string", "minLength": 1},
            "project_id": {"type": "string", "minLength": 1},
            "filename": {"type": "string", "minLength": 1},
            "content_type": {"type": "string", "minLength": 1},
        },
        "required": [],
    }
    requires_approval = False
    resource_key_prefix = "artifact"
    resource_target_default = "catalog"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        filters = {
            key: value for key, value in arguments.items() if value not in (None, "")
        }
        return runtime.invoke(self.host_method, filters)


class SaveArtifactTool(Tool):
    """Register an existing workspace file as a versioned Artifact."""

    name = "save_artifact"
    host_method = "save_artifact"
    description = "Register an existing workspace file as a versioned artifact."
    parameters = {
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Existing workspace file to register.",
            },
            "filename": {"type": "string", "minLength": 1},
            "content_type": {"type": "string", "minLength": 1},
            "input_version_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 500,
            },
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["path"],
    }
    read_only = False
    permission_target_key = "path"
    secret_path_key = "path"
    side_effect_class = "external_write"
    resource_key_prefix = "artifact"
    resource_target_key = "path"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, dict(arguments))


class GetArtifactMetadataTool(Tool):
    """Read exact, path-free metadata for one Artifact/version identity."""

    name = "get_artifact_metadata"
    host_method = "get_artifact_metadata"
    description = (
        "Read exact metadata for one session artifact and its latest or specified "
        "immutable version."
    )
    parameters = {
        "properties": {
            "artifact_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "version_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Omit to inspect the current Artifact version.",
            },
        },
        "required": ["artifact_id"],
    }
    requires_approval = False
    resource_key_prefix = "artifact"
    resource_target_key = "artifact_id"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        spec = arguments if isinstance(arguments, dict) else {}
        keys = [resource_key("artifact", spec.get("artifact_id"))]
        if spec.get("version_id"):
            keys.append(resource_key("artifact_version", spec["version_id"]))
        return tuple(keys)

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, dict(arguments))


class ListArtifactVersionsTool(Tool):
    """List the immutable version identities for one exact Artifact."""

    name = "list_artifact_versions"
    host_method = "list_artifact_versions"
    description = "List immutable versions for one session-owned artifact."
    parameters = {
        "properties": {
            "artifact_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            }
        },
        "required": ["artifact_id"],
    }
    requires_approval = False
    resource_key_prefix = "artifact"
    resource_target_key = "artifact_id"

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, dict(arguments))


class RestoreArtifactVersionTool(Tool):
    """Restore verified historical bytes as a fresh immutable version."""

    name = "restore_artifact_version"
    host_method = "restore_artifact_version"
    description = (
        "Restore a historical immutable artifact snapshot into the workspace as "
        "a new version, preserving all prior versions and lineage."
    )
    parameters = {
        "properties": {
            "artifact_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "version_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Historical source version to copy and restore.",
            },
        },
        "required": ["artifact_id", "version_id"],
    }
    read_only = False
    # The focused service records the restored bytes itself. Marking this as a
    # generic file writer would make the Web boundary capture a duplicate
    # Artifact version after the tool returns.
    writes_files = False
    dangerous = True
    requires_approval = True
    side_effect_class = "high_risk"
    resource_key_prefix = "artifact"
    resource_target_key = "artifact_id"

    def permission_target(self, arguments: Any) -> str:
        spec = arguments if isinstance(arguments, dict) else {}
        artifact_id = str(spec.get("artifact_id") or "")
        version_id = str(spec.get("version_id") or "")
        return f"{artifact_id}@{version_id}"

    def resource_keys(self, arguments: Any) -> tuple[str, ...]:
        spec = arguments if isinstance(arguments, dict) else {}
        return (
            resource_key("artifact", spec.get("artifact_id")),
            resource_key("artifact_version", spec.get("version_id")),
            resource_key("workspace", spec.get("artifact_id")),
        )

    def execute(self, runtime: ControlToolContext, arguments: dict) -> dict:
        return runtime.invoke(self.host_method, dict(arguments))


__all__ = [
    "GetArtifactMetadataTool",
    "ListArtifactsTool",
    "ListArtifactVersionsTool",
    "RestoreArtifactVersionTool",
    "SaveArtifactTool",
]
