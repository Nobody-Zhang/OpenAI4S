"""Offline MCP protocol and child-environment contracts."""

from __future__ import annotations

import os

import pytest

from openai4s import mcp_client
from openai4s.mcp_client import (
    MCPConnection,
    MCPError,
    MCPManager,
    _connector_environment,
    example_server_config,
)
from openai4s.mcp_servers.example_server import RESOURCE_URI


def test_connector_environment_is_allowlisted_and_explicit_env_is_the_secret_boundary():
    source = {
        "PATH": "/safe/bin",
        "HOME": "/safe/home",
        "LANG": "en_US.UTF-8",
        "OPENAI4S_LLM_API_KEY": "daemon-provider-secret",
        "AWS_SECRET_ACCESS_KEY": "daemon-cloud-secret",
        "HTTP_PROXY": "https://user:password@proxy.invalid",
        "PYTHONPATH": "/untrusted/imports",
        "NODE_OPTIONS": "--require=/untrusted/bootstrap.js",
    }

    env = _connector_environment(
        {"SCIENCE_MCP_TOKEN": "connector-secret", "MODE": 7},
        source=source,
    )

    assert env["PATH"] == "/safe/bin"
    assert env["HOME"] == "/safe/home"
    assert env["LANG"] == "en_US.UTF-8"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["SCIENCE_MCP_TOKEN"] == "connector-secret"
    assert env["MODE"] == "7"
    assert set(env).isdisjoint(
        {
            "OPENAI4S_LLM_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "HTTP_PROXY",
            "PYTHONPATH",
            "NODE_OPTIONS",
        }
    )


def test_connector_environment_has_a_path_fallback_and_rejects_invalid_entries():
    assert _connector_environment(source={})["PATH"] == os.defpath

    with pytest.raises(MCPError, match="must be an object"):
        _connector_environment([("TOKEN", "value")], source={})
    with pytest.raises(MCPError, match="invalid connector env name"):
        _connector_environment({"BAD=NAME": "value"}, source={})
    with pytest.raises(MCPError, match="cannot be null"):
        _connector_environment({"TOKEN": None}, source={})
    with pytest.raises(MCPError, match="contains NUL"):
        _connector_environment({"TOKEN": "bad\x00value"}, source={})


def test_manager_connect_never_passes_ambient_secrets_to_popen(monkeypatch):
    captured = {}

    class CapturingConnection:
        def __init__(self, command, env=None, cwd=None):
            captured.update(command=command, env=env, cwd=cwd)

    monkeypatch.setenv("OPENAI4S_LLM_API_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-cloud-secret")
    monkeypatch.setattr(mcp_client, "MCPConnection", CapturingConnection)

    manager = MCPManager()
    connection = manager._connect(
        {
            "command": ["science-mcp"],
            "args": ["--stdio"],
            "env": {"SCIENCE_MCP_TOKEN": "declared-secret"},
            "cwd": "/connector/workspace",
        }
    )

    assert isinstance(connection, CapturingConnection)
    assert captured["command"] == ["science-mcp", "--stdio"]
    assert captured["cwd"] == "/connector/workspace"
    assert captured["env"]["SCIENCE_MCP_TOKEN"] == "declared-secret"
    assert "OPENAI4S_LLM_API_KEY" not in captured["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]


def test_connection_uses_standard_resource_and_prompt_method_shapes():
    connection = object.__new__(MCPConnection)
    calls: list[tuple[str, dict | None]] = []

    def request(method, params=None):
        calls.append((method, params))
        return {
            "resources/list": {"resources": [], "nextCursor": "r-next"},
            "resources/read": {"contents": []},
            "prompts/list": {"prompts": [], "nextCursor": "p-next"},
            "prompts/get": {"messages": []},
        }[method]

    connection._request = request

    assert connection.list_resources("r-1")["nextCursor"] == "r-next"
    assert connection.read_resource("science://dataset") == {"contents": []}
    assert connection.list_prompts("p-1")["nextCursor"] == "p-next"
    assert connection.get_prompt("analyze", {"kind": "fast"}) == {"messages": []}
    assert calls == [
        ("resources/list", {"cursor": "r-1"}),
        ("resources/read", {"uri": "science://dataset"}),
        ("prompts/list", {"cursor": "p-1"}),
        (
            "prompts/get",
            {"name": "analyze", "arguments": {"kind": "fast"}},
        ),
    ]


def test_bundled_server_supports_resources_and_prompts_end_to_end():
    manager = MCPManager()
    config = example_server_config()
    try:
        resources = manager.list_resources("example", config)
        assert resources["resources"][0]["uri"] == RESOURCE_URI

        content = manager.read_resource("example", config, RESOURCE_URI)
        assert content["contents"][0]["uri"] == RESOURCE_URI
        assert "third-party packages" in content["contents"][0]["text"]

        prompts = manager.list_prompts("example", config)
        assert prompts["prompts"][0]["name"] == "summarize"

        rendered = manager.get_prompt(
            "example",
            config,
            "summarize",
            {"text": "alpha beta gamma"},
        )
        message = rendered["messages"][0]
        assert message["role"] == "user"
        assert "alpha beta gamma" in message["content"]["text"]
    finally:
        manager.shutdown()
