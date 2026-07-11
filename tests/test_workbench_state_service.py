from __future__ import annotations

from types import SimpleNamespace

from openai4s.config import LLMConfig
from openai4s.server.workbench_state import SessionWorkbenchStateService


class _Store:
    def get_frame(self, frame_id):
        return {"frame_id": frame_id, "root_frame_id": frame_id}


def _service(state=None, pending=()):
    return SessionWorkbenchStateService(
        _Store(),
        state_for=lambda _root: state,
        history_for=lambda _root: [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world", "compaction_handoff": True},
        ],
        llm_config_for=lambda _state: LLMConfig(
            provider="deepseek", model="deepseek-chat", api_key="test"
        ),
        pending_for=lambda _root: pending,
        context_window_fallback=10_000,
    )


def test_context_projects_components_without_message_content():
    result = _service().context("root")
    assert result["token_count"] > 0
    assert result["handoff"] is True
    assert {item["kind"] for item in result["layers"]} == {
        "text",
        "images",
        "tool_calls",
        "wire_state",
    }
    assert "hello" not in repr(result)


def test_security_never_claims_unstarted_sandbox(monkeypatch):
    monkeypatch.setenv("OPENAI4S_KERNEL_SANDBOX", "enforce")
    result = _service(pending=({"decision_id": "secret"},)).security("root")
    assert result["sandbox"]["state"] == "not_started"
    assert result["sandbox"]["enforced"] is False
    assert result["permission"]["pending_count"] == 1
    assert "secret" not in repr(result)


def test_security_uses_only_public_live_sandbox_fields():
    kernel = SimpleNamespace(
        sandbox_status={
            "mode": "auto",
            "state": "enforced",
            "backend": "seatbelt",
            "enforced": True,
            "self_test_passed": True,
            "network_policy": "blocked",
            "workspace": "/private/session",
            "temp_dir": "/private/tmp",
            "detail": "verified",
        }
    )
    state = SimpleNamespace(kernel=kernel)
    result = _service(state=state).security("root")
    assert result["sandbox"]["enforced"] is True
    assert result["sandbox"]["backend"] == "seatbelt"
    assert "workspace" not in result["sandbox"]
    assert "/private" not in repr(result)
