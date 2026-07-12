"""Delegated child execution policy is enforced, not decorative metadata."""

from __future__ import annotations

import pytest

import openai4s.agent.loop as loop_mod
from openai4s.agent.delegation import DelegationError, DelegationRunner
from openai4s.config import get_config


def _submitted(output=None):
    return {
        "stop_reason": "submitted",
        "submitted_output": {
            "output": output if output is not None else {"ok": True},
            "completion_bullets": ["child complete"],
        },
        "final_message": None,
    }


def test_child_policy_filters_catalog_and_direct_host_calls(monkeypatch):
    observed = {}

    def inspect_policy(self, task):
        del task
        observed["tools"] = {
            tool.name for tool in self.dispatcher.tool_catalog().tools()
        }
        observed["write"] = self.dispatcher(
            "write_file", [{"path": "blocked.txt", "content": "blocked"}]
        )
        observed["bash"] = self.dispatcher(
            "authorize_bash",
            [
                {
                    "command": "echo blocked",
                    "command_sha256": "0" * 64,
                    "cwd": ".",
                    "generation": "test",
                    "challenge": "test",
                }
            ],
        )
        observed["capabilities"] = self.dispatcher("capabilities", [])
        observed["max_turns"] = self.max_turns
        observed["allow_delegate"] = self.allow_delegate
        observed["skill_loader"] = self._skill_loader
        return _submitted()

    monkeypatch.setattr(loop_mod.Agent, "run", inspect_policy)
    result = DelegationRunner(get_config())(
        {
            "request": "read and browse only",
            "steps": 2,
            "capabilities": ["web", "read_file", "bash"],
            "permissions": {"bash": "deny"},
        }
    )

    assert result["output"] == {"ok": True}
    assert {"web_search", "web_fetch", "read_text_file"} <= observed["tools"]
    assert "write_file" not in observed["tools"]
    assert observed["write"] == {
        "error": "Capability denied by delegated child policy: write_file"
    }
    assert observed["bash"] == {
        "error": "Permission denied by delegated child policy: authorize_bash"
    }
    assert observed["capabilities"]["bash"] is False
    assert observed["capabilities"]["web_search"] is True
    assert observed["max_turns"] == 2
    assert observed["allow_delegate"] is False
    assert observed["skill_loader"] is None


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"permissions": {"write_file": "sometimes"}}, "allow, ask, or deny"),
        ({"steps": 0}, "positive integer"),
    ],
)
def test_invalid_policy_fails_before_consuming_budget(override, fragment):
    runner = DelegationRunner(get_config())
    with pytest.raises(DelegationError, match=fragment):
        runner({"request": "invalid", **override})
    assert runner.delegation_stats()["spawned_session"] == 0


def test_nested_child_cannot_widen_parent_capabilities(monkeypatch):
    observed = {}

    def nested_attempt(self, task):
        del task
        try:
            self.dispatcher._delegate_fn(
                {"request": "try to write", "capabilities": ["write_file"]}
            )
        except DelegationError as error:
            observed["error"] = str(error)
        return _submitted()

    monkeypatch.setattr(loop_mod.Agent, "run", nested_attempt)
    runner = DelegationRunner(get_config())
    runner(
        {
            "request": "parent",
            "capabilities": ["delegation", "read_file"],
        }
    )
    assert "exceed parent policy: write_file" in observed["error"]
    assert runner.delegation_stats()["spawned_session"] == 1


def test_nested_permission_deny_cannot_be_relaxed(monkeypatch):
    observed = {}

    def nested_policy(self, task):
        del task
        if self.delegate_depth == 1:
            return _submitted(
                self.dispatcher._delegate_fn(
                    {
                        "request": "nested",
                        "permissions": {"bash": "allow"},
                    }
                )
            )
        observed["bash"] = self.dispatcher(
            "authorize_bash",
            [
                {
                    "command": "echo blocked",
                    "command_sha256": "0" * 64,
                    "cwd": ".",
                    "generation": "test",
                    "challenge": "test",
                }
            ],
        )
        return _submitted({"checked": True})

    monkeypatch.setattr(loop_mod.Agent, "run", nested_policy)
    runner = DelegationRunner(get_config())
    result = runner(
        {
            "request": "parent",
            "capabilities": ["delegation", "bash"],
            "permissions": {"bash": "deny"},
        }
    )
    assert result["output"]["output"] == {"checked": True}
    assert observed["bash"] == {
        "error": "Permission denied by delegated child policy: authorize_bash"
    }
    descendants = runner._tree.descendants(
        runner.children()[0]["child_id"], include_self=False
    )
    assert descendants[0].snapshot()["overrides"]["permissions"]["bash"] == "deny"
