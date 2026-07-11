"""Single-URL web-fetch control tool."""

from __future__ import annotations

import re
from typing import Any

from openai4s.tools.base import Tool


class WebFetchTool(Tool):
    """Normalize fetch arguments and preserve the host soft-fail contract."""

    name = "web_fetch"
    host_method = "web_fetch"
    description = "Fetch a URL and return its content as markdown/text/html/json."
    parameters = {
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
            "format": {
                "type": "string",
                "description": "One of markdown|text|html|json (default markdown).",
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate the returned content to this many characters.",
            },
            "timeout": {
                "type": "number",
                "description": "Seconds to wait before giving up (default 30).",
            },
        },
        "required": ["url"],
    }
    needs_network = True
    screen_untrusted_output = True

    def permission_target(self, arguments: Any) -> str:
        if not isinstance(arguments, dict):
            return ""
        url = str(arguments.get("url") or "")
        return re.sub(r"^https?://(www\.)?", "", url).split("/")[0]

    def execute(self, _runtime: Any, arguments: dict) -> dict:
        from openai4s import egress, webtools

        try:
            return webtools.web_fetch(
                arguments.get("url", ""),
                fmt=arguments.get("format", "markdown"),
                timeout=float(arguments.get("timeout") or 30),
                max_chars=int(arguments.get("max_chars") or 20000),
            )
        except (webtools.NetworkDisabled, egress.EgressBlocked) as error:
            return {"error": str(error)}
        except Exception as error:  # noqa: BLE001
            return {"error": f"web_fetch: {error}"}


__all__ = ["WebFetchTool"]
