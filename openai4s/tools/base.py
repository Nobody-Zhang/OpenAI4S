"""Executable base class for the agent's control-plane tools.

Concrete tools declare their metadata as class attributes and keep their
domain behaviour in ``execute``.  Model-originated calls must enter through
``invoke`` so the :class:`HostDispatcher` can apply permissions, approvals,
auditing, injection screening, and UI activity events before ``execute`` is
reached by the dispatcher's thin host-method adapter.

The positional constructor remains available for callers that build dynamic
metadata-only tools. OpenAI4S built-ins use named subclasses, mirroring
CoreCoder's extensible tool catalogue.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from typing import Any


class Tool:
    """Base interface and compatibility adapter for one control tool.

    Subclasses normally set the metadata below as class attributes and
    override :meth:`execute`.  ``Tool(name, host_method, ...)`` is retained for
    schema-only extensions and compatibility; built-in tools never use it.
    Concrete context ports are documented in :mod:`openai4s.tools.contexts`.
    """

    name: str = ""
    host_method: str = ""
    description: str = ""
    parameters: dict = {"properties": {}, "required": []}
    read_only: bool = True
    writes_files: bool = False
    needs_network: bool = False
    mutates_cwd: bool = False
    dangerous: bool = False
    output_limit: int = 20_000
    requires_approval: bool = True
    permission_target_key: str | None = None
    permission_target_default: str = ""
    secret_path_key: str | None = None
    screen_untrusted_output: bool = False

    _METADATA_FIELDS = (
        "name",
        "host_method",
        "description",
        "parameters",
        "read_only",
        "writes_files",
        "needs_network",
        "mutates_cwd",
        "dangerous",
        "output_limit",
        "requires_approval",
        "permission_target_key",
        "permission_target_default",
        "secret_path_key",
        "screen_untrusted_output",
    )

    def __init__(
        self,
        name: str | None = None,
        host_method: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        read_only: bool | None = None,
        writes_files: bool | None = None,
        needs_network: bool | None = None,
        mutates_cwd: bool | None = None,
        dangerous: bool | None = None,
        output_limit: int | None = None,
        requires_approval: bool | None = None,
        permission_target_key: str | None = None,
        permission_target_default: str | None = None,
        secret_path_key: str | None = None,
        screen_untrusted_output: bool | None = None,
    ) -> None:
        # No arguments snapshots the concrete subclass's class declarations.
        overrides = {
            "name": name,
            "host_method": host_method,
            "description": description,
            "parameters": parameters,
            "read_only": read_only,
            "writes_files": writes_files,
            "needs_network": needs_network,
            "mutates_cwd": mutates_cwd,
            "dangerous": dangerous,
            "output_limit": output_limit,
            "requires_approval": requires_approval,
            "permission_target_key": permission_target_key,
            "permission_target_default": permission_target_default,
            "secret_path_key": secret_path_key,
            "screen_untrusted_output": screen_untrusted_output,
        }
        for field, value in overrides.items():
            if value is None:
                value = getattr(type(self), field)
            if field == "parameters":
                value = copy.deepcopy(value)
            object.__setattr__(self, field, value)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        values = ", ".join(
            f"{field}={getattr(self, field)!r}" for field in self._METADATA_FIELDS
        )
        return f"{type(self).__name__}({values})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return False
        return all(
            getattr(self, field) == getattr(other, field)
            for field in self._METADATA_FIELDS
        )

    def invoke(self, dispatcher: Any, arguments: dict) -> Any:
        """Enter the host's policy envelope for a model-originated call.

        This is deliberately separate from :meth:`execute`: direct execution
        is reserved for the HostDispatcher's protected method adapters.
        """
        return dispatcher(self.host_method, [dict(arguments)])

    def execute(self, context: Any, arguments: dict) -> Any:
        """Run the tool's domain behaviour inside the dispatcher envelope."""
        raise NotImplementedError(f"{type(self).__name__} does not implement execute()")

    def native_precheck(self, arguments: dict) -> str | None:
        """Return a cheap pre-dispatch error for native/fenced calls, if any."""
        return None

    def permission_target(self, arguments: Any) -> str:
        """Return the value matched by the permission broker.

        The secure default is the tool name. Concrete tools can declare a
        simple argument key or override this method for derived targets.
        """
        if self.permission_target_key and isinstance(arguments, dict):
            value = arguments.get(self.permission_target_key)
            if value not in (None, ""):
                return str(value)
            return self.permission_target_default
        if isinstance(arguments, str) and arguments:
            return arguments
        return self.permission_target_default or self.name

    def secret_path(self, arguments: Any) -> str | None:
        """Return a direct file target requiring the hard secret denylist."""
        if not self.secret_path_key or not isinstance(arguments, dict):
            return None
        value = arguments.get(self.secret_path_key)
        return str(value or "")

    def signature_line(self) -> str:
        """Return ``name(arg1, arg2?, ...)`` using declared parameter order."""
        props = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        parts = [(arg if arg in required else f"{arg}?") for arg in props]
        return f"{self.name}({', '.join(parts)})"

    def schema(self) -> dict:
        """Return an OpenAI-style function schema for simple integrations."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": copy.deepcopy(
                        self.parameters.get("properties") or {}
                    ),
                    "required": list(self.parameters.get("required") or []),
                },
            },
        }


__all__ = ["Tool"]
