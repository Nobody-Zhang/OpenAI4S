"""Minimal MCP (Model Context Protocol) stdio client — pure stdlib.

Speaks newline-delimited JSON-RPC 2.0 to a spawned MCP server process, enough to
power the Connectors control plane: handshake (initialize + initialized), tools,
resources, and prompts. A process-wide MCPManager caches one live connection per
connector id so repeated calls reuse the same server.

Sampling and server-initiated requests are deliberately outside this client.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
_DEFAULT_TIMEOUT = 30.0

# A connector is third-party code.  Never copy the daemon's complete environment
# into it: that would silently expose provider keys, cloud credentials, and
# unrelated application secrets.  These variables are the small cross-platform
# runtime substrate needed to locate a command, create temporary files, select a
# locale, and use the host trust store.  Connector-specific credentials must be
# supplied explicitly in the persisted connector ``env`` mapping.
_CONNECTOR_RUNTIME_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
)


class MCPError(RuntimeError):
    pass


def _connector_environment(
    explicit: Mapping[str, Any] | None = None,
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a fresh least-privilege environment for one MCP subprocess.

    ``source`` exists for deterministic tests.  Explicit connector values are
    intentionally allowed to contain credentials: they are the connector's
    declared secret boundary, unlike arbitrary variables inherited by the
    daemon from the user's shell.
    """

    host = os.environ if source is None else source
    env = {
        name: str(host[name])
        for name in _CONNECTOR_RUNTIME_ENV
        if name in host and host[name] is not None
    }
    env.setdefault("PATH", os.defpath)
    env["PYTHONUNBUFFERED"] = "1"
    if explicit is None:
        return env
    if not isinstance(explicit, Mapping):
        raise MCPError("connector env must be an object")
    for raw_name, raw_value in explicit.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise MCPError("connector env names must be non-empty strings")
        if "=" in raw_name or "\x00" in raw_name:
            raise MCPError(f"invalid connector env name: {raw_name!r}")
        if raw_value is None:
            raise MCPError(f"connector env value for {raw_name!r} cannot be null")
        value = str(raw_value)
        if "\x00" in value:
            raise MCPError(f"connector env value for {raw_name!r} contains NUL")
        env[raw_name] = value
    return env


class MCPConnection:
    def __init__(
        self, command: list[str], env: dict | None = None, cwd: str | None = None
    ):
        self.command = command
        self._id = 0
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        self._init()

    # -- wire ----------------------------------------------------------------
    def _send(self, obj: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read_reply(self, want_id: int) -> dict:
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise MCPError("MCP server closed the connection")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # skip notifications / unrelated ids
            if msg.get("id") != want_id:
                continue
            if "error" in msg and msg["error"] is not None:
                raise MCPError(str(msg["error"].get("message") or msg["error"]))
            return msg.get("result") or {}

    def _request(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            self._id += 1
            mid = self._id
            self._send(
                {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
            )
            return self._read_reply(mid)

    def _notify(self, method: str, params: dict | None = None) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- lifecycle -----------------------------------------------------------
    def _init(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "openai4s", "version": "1.0.0"},
            },
        )
        try:
            self._notify("notifications/initialized")
        except Exception:  # noqa: BLE001
            pass

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- tools ---------------------------------------------------------------
    def list_tools(self) -> list[dict]:
        res = self._request("tools/list")
        return res.get("tools", []) if isinstance(res, dict) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        res = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        # normalize content blocks -> plain text for the agent
        text_parts = []
        for block in res.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return {
            "is_error": bool(res.get("isError")),
            "text": "\n".join(text_parts),
            "raw": res,
        }

    # -- resources -----------------------------------------------------------
    def list_resources(self, cursor: str | None = None) -> dict:
        params = {"cursor": cursor} if cursor is not None else None
        res = self._request("resources/list", params)
        if not isinstance(res, dict):
            raise MCPError("resources/list returned a non-object result")
        return res

    def read_resource(self, uri: str) -> dict:
        res = self._request("resources/read", {"uri": uri})
        if not isinstance(res, dict):
            raise MCPError("resources/read returned a non-object result")
        return res

    # -- prompts -------------------------------------------------------------
    def list_prompts(self, cursor: str | None = None) -> dict:
        params = {"cursor": cursor} if cursor is not None else None
        res = self._request("prompts/list", params)
        if not isinstance(res, dict):
            raise MCPError("prompts/list returned a non-object result")
        return res

    def get_prompt(self, name: str, arguments: dict | None = None) -> dict:
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        res = self._request("prompts/get", params)
        if not isinstance(res, dict):
            raise MCPError("prompts/get returned a non-object result")
        return res


class MCPManager:
    """One live connection per connector id (lazily connected, cached)."""

    def __init__(self) -> None:
        self._conns: dict[str, MCPConnection] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _argv(config: dict) -> list[str]:
        cmd = config.get("command")
        args = config.get("args") or []
        if isinstance(cmd, list):
            argv = list(cmd) + list(args)
        elif isinstance(cmd, str) and cmd.strip():
            argv = cmd.split() + list(args)
        else:
            raise MCPError("connector has no command")
        return argv

    def _connect(self, config: dict) -> MCPConnection:
        env = _connector_environment(config.get("env"))
        return MCPConnection(self._argv(config), env=env, cwd=config.get("cwd"))

    def get(self, connector_id: str, config: dict) -> MCPConnection:
        with self._lock:
            conn = self._conns.get(connector_id)
            if conn is not None and conn.alive():
                return conn
            if conn is not None:
                conn.close()
            conn = self._connect(config)
            self._conns[connector_id] = conn
            return conn

    def probe(self, config: dict) -> dict:
        """Connect fresh, list tools, close. Returns {ok, tools|error}."""
        try:
            conn = self._connect(config)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        try:
            tools = conn.list_tools()
            return {"ok": True, "tools": tools}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()

    def list_tools(self, connector_id: str, config: dict) -> list[dict]:
        return self.get(connector_id, config).list_tools()

    def call_tool(
        self, connector_id: str, config: dict, tool: str, arguments: dict | None = None
    ) -> dict:
        return self.get(connector_id, config).call_tool(tool, arguments)

    def list_resources(
        self,
        connector_id: str,
        config: dict,
        cursor: str | None = None,
    ) -> dict:
        return self.get(connector_id, config).list_resources(cursor)

    def read_resource(self, connector_id: str, config: dict, uri: str) -> dict:
        return self.get(connector_id, config).read_resource(uri)

    def list_prompts(
        self,
        connector_id: str,
        config: dict,
        cursor: str | None = None,
    ) -> dict:
        return self.get(connector_id, config).list_prompts(cursor)

    def get_prompt(
        self,
        connector_id: str,
        config: dict,
        name: str,
        arguments: dict | None = None,
    ) -> dict:
        return self.get(connector_id, config).get_prompt(name, arguments)

    def disconnect(self, connector_id: str) -> None:
        with self._lock:
            conn = self._conns.pop(connector_id, None)
        if conn is not None:
            conn.close()

    def shutdown(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for c in conns:
            c.close()


# a process-wide manager (the daemon is single-process)
_MANAGER: MCPManager | None = None


def manager() -> MCPManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = MCPManager()
    return _MANAGER


def example_server_config() -> dict:
    """Config for the bundled example server (always available)."""
    return {"command": [sys.executable, "-m", "openai4s.mcp_servers.example_server"]}
