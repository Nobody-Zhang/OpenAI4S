"""Runtime package-installation control tool."""

from __future__ import annotations

from typing import Any

from openai4s.tools.base import Tool


class EnvCreateTool(Tool):
    """Install requested packages through the kernel preinstall service."""

    name = "env_create"
    host_method = "env_setup"
    description = "Install extra packages into the current kernel (pip)."
    parameters = {
        "properties": {
            "name": {
                "type": "string",
                "description": "Optional environment label.",
            },
            "packages": {
                "type": "array",
                "description": "Package names to install.",
            },
        },
        "required": ["packages"],
    }
    read_only = False

    def execute(self, _runtime: Any, arguments: dict) -> dict:
        from openai4s.kernel import preinstall

        arguments = arguments or {}
        packages = [
            package
            for package in (arguments.get("packages") or [])
            if isinstance(package, str)
        ]
        name = arguments.get("name") or "analysis"
        if not packages:
            return {
                "name": name,
                "installed": [],
                "ok": True,
                "note": "no packages requested",
            }
        result = preinstall.install(packages)
        result["name"] = name
        return result


__all__ = ["EnvCreateTool"]
