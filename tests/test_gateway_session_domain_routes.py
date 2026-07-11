from __future__ import annotations

import hashlib
import platform

import pytest

from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod


class _Hub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emitter(self, root_frame_id):
        def emit(event):
            event.setdefault("root_frame_id", root_frame_id)
            self.events.append(event)

        return emit

    def broadcast(self, root_frame_id, event):
        self.emitter(root_frame_id)(event)

    def has_subscriber(self, root_frame_id):
        del root_frame_id
        return False


def _setup(tmp_path):
    config = Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
    )
    hub = _Hub()
    runner = gateway_mod.SessionRunner(config, hub, start_idle_sweeper=False)
    frame_id = runner.store.new_frame(
        kind="turn", project_id="project-domain", status="ready"
    )
    handler_class = gateway_mod.make_handler(config, hub, runner)
    handler = object.__new__(handler_class)
    return runner, handler, frame_id


def _call(handler, method, path, *, body=None, query=None):
    replies = []
    handler._query = lambda: query or {}
    handler._body = lambda: body or {}
    handler._json = lambda value, code=200: replies.append((code, value))
    handler._send = lambda code, data, content_type, extra=None: replies.append(
        (code, data, content_type, extra or {})
    )
    handler._api(method, path)
    assert replies
    return replies[-1]


def test_checkpoint_fork_and_workbench_routes_share_domain_service(tmp_path):
    runner, handler, frame_id = _setup(tmp_path)
    workspace = runner.workspace_for(frame_id)
    (workspace / "analysis.txt").write_text("checkpointed\n", "utf-8")

    code, branches = _call(handler, "GET", f"/frames/{frame_id}/branches")
    assert code == 200
    assert branches["current_branch_id"] == frame_id
    assert branches["capabilities"]["checkpoint"]["enabled"] is True

    code, checkpoint = _call(
        handler,
        "POST",
        f"/frames/{frame_id}/branches/checkpoints",
        body={"reason": "browser"},
    )
    assert code == 200
    checkpoint_id = checkpoint["checkpoint_id"]

    code, forked = _call(
        handler,
        "POST",
        f"/frames/{frame_id}/branches/fork",
        body={"from_checkpoint_id": checkpoint_id, "name": "alternative"},
    )
    assert code == 200
    branch_id = forked["branch_id"]
    branch_workspace = runner.workspace_for_branch(frame_id, branch_id)
    assert branch_workspace != workspace
    assert (branch_workspace / "analysis.txt").read_text("utf-8") == "checkpointed\n"

    code, timeline = _call(
        handler, "GET", f"/frames/{frame_id}/action-timeline"
    )
    assert code == 200
    assert "checkpoint" in {group["kind"] for group in timeline["groups"]}
    code, branch_timeline = _call(
        handler,
        "GET",
        f"/frames/{frame_id}/action-timeline",
        query={"branch_id": [branch_id]},
    )
    assert code == 200
    assert "branch" in {group["kind"] for group in branch_timeline["groups"]}
    assert "canonical_arguments" not in repr(timeline)

    code, context = _call(handler, "GET", f"/frames/{frame_id}/context")
    assert code == 200 and "layers" in context
    code, security = _call(handler, "GET", f"/frames/{frame_id}/security")
    assert code == 200 and security["sandbox"]["state"] == "not_started"
    code, recovery = _call(handler, "GET", f"/frames/{frame_id}/recovery")
    assert code == 200 and recovery["root_frame_id"] == frame_id
    runner.close()


def test_notebook_export_and_renderer_routes_return_immutable_descriptors(tmp_path):
    runner, handler, frame_id = _setup(tmp_path)
    code, error = _call(
        handler,
        "GET",
        f"/frames/{frame_id}/notebook/export",
        query={"language": ["javascript"]},
    )
    assert code == 400
    assert error == {"error": "notebook language must be python, r, or bundle"}

    binary = _call(
        handler,
        "GET",
        f"/frames/{frame_id}/notebook/export",
        query={"language": ["python"]},
    )
    assert binary[0] == 200
    assert binary[2] == "application/x-ipynb+json"
    assert b'"nbformat": 4' in binary[1]
    assert len(binary[3]["X-Content-SHA256"]) == 64

    image = runner.workspace_for(frame_id) / "plot.png"
    image.write_bytes(b"not-a-rendered-image")
    artifact = runner.store.save_artifact(
        path=str(image),
        filename="plot.png",
        content_type="image/png",
        size_bytes=image.stat().st_size,
        checksum=hashlib.sha256(image.read_bytes()).hexdigest(),
        frame_id=frame_id,
        root_frame_id=frame_id,
        project_id="project-domain",
    )
    code, descriptor = _call(
        handler,
        "GET",
        f"/artifacts/{artifact['artifact_id']}/renderer",
        query={"root_frame_id": [frame_id]},
    )
    assert code == 200
    assert descriptor["renderer"]["renderer_id"] == "image"
    stored = runner.store.get_artifact(artifact["artifact_id"])
    assert descriptor["version_id"] == stored["latest_version_id"]
    assert descriptor["trusted_html"] is False
    runner.close()


def test_unknown_session_workbench_routes_fail_with_one_404_contract(tmp_path):
    runner, handler, _frame_id = _setup(tmp_path)
    routes = (
        "/frames/missing/action-timeline",
        "/frames/missing/execution-queue",
        "/frames/missing/context",
        "/frames/missing/security",
        "/frames/missing/recovery",
        "/frames/missing/recovery/actions",
        "/frames/missing/branches",
        "/frames/missing/checkpoints",
        "/frames/missing/branches/checkpoints",
        "/frames/missing/revert/operations",
        "/frames/missing/notebook/export",
        "/frames/missing/execution",
    )
    try:
        for route in routes:
            with pytest.raises(gateway_mod.GatewayError) as caught:
                _call(handler, "GET", route)
            assert caught.value.code == 404
            assert caught.value.message == "session not found"
    finally:
        runner.close()


def test_restart_permission_route_requires_explicit_continuation(tmp_path):
    runner, _handler, frame_id = _setup(tmp_path)
    payload = {
        "type": "await_permission",
        "frame_id": frame_id,
        "decision_id": "perm-route-restart",
        "tool": "mcp_call",
        "target": "lab/send",
    }
    runner.store.append_tool_action_group(
        root_frame_id=frame_id,
        turn_id="turn-before-restart",
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-before-restart",
                    "name": "mcp_call",
                    "arguments": {"server": "lab", "tool": "send"},
                }
            ],
        },
        events=[
            {
                "type": "proposed",
                "tool_call_id": "call-before-restart",
                "canonical_arguments": {
                    "name": "mcp_call",
                    "arguments": {"server": "lab", "tool": "send"},
                },
            }
        ],
    )
    runner.store.create_permission_request(
        decision_id="perm-route-restart",
        root_frame_id=frame_id,
        frame_id=frame_id,
        project_id="project-domain",
        tool="mcp_call",
        target="lab/send",
        payload=payload,
    )
    runner.close()
    runner.store.close()

    config = Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
    )
    hub = _Hub()
    restarted = gateway_mod.SessionRunner(
        config, hub, start_idle_sweeper=False
    )
    handler_class = gateway_mod.make_handler(config, hub, restarted)
    handler = object.__new__(handler_class)
    try:
        assert restarted._sessions == {}
        code, security = _call(
            handler, "GET", f"/frames/{frame_id}/security"
        )
        assert code == 200
        assert security["permission"]["pending_count"] == 1

        code, resolution = _call(
            handler,
            "POST",
            f"/frames/{frame_id}/decision",
            body={
                "decision_id": "perm-route-restart",
                "allow": True,
                "scope": "once",
            },
        )
        assert code == 200
        assert resolution["ok"] is True
        assert resolution["decision_id"] == "perm-route-restart"
        assert resolution["resolution_context"] == "after_restart"
        assert resolution["requires_continue"] is True
        assert resolution["original_action_executed"] is False
        assert resolution["continuation_authorization"] == "once"
        assert resolution["continuation_expires_at"] is not None
        request = restarted.store.get_permission_request("perm-route-restart")
        assert request["state"] == "allowed"
        assert request["continuation_consumed_at"] is None
        marker = restarted.store.list_action_groups(frame_id)[-1]
        assert marker["kind"] == "permission_resolution"
        assert "arguments" not in repr(marker["events"][0]["result"])

        events = [
            event
            for event in hub.events
            if event.get("type") == "permission_resolved"
        ]
        assert len(events) == 1
        assert events[0]["requires_continue"] is True
        assert events[0]["original_action_executed"] is False
    finally:
        restarted.close()
        restarted.store.close()


def test_fork_from_cell_route_fails_closed_until_supported(tmp_path):
    runner, handler, frame_id = _setup(tmp_path)
    try:
        _call(
            handler,
            "POST",
            f"/frames/{frame_id}/branches/fork",
            body={"from_cell_id": "cell-1"},
        )
    except gateway_mod.GatewayError as error:
        assert error.code == 409
        assert "checkpoint" in error.message
    else:
        raise AssertionError("fork-from-cell must not claim success")
    runner.close()


def test_real_runner_checkpoint_can_restore_through_mutation_route(tmp_path):
    runner, handler, frame_id = _setup(tmp_path)
    workspace = runner.workspace_for(frame_id)
    (workspace / "analysis.txt").write_text("checkpoint bytes\n", "utf-8")
    nested = workspace / "results" / "out.csv"
    nested.parent.mkdir()
    nested.write_text("score\n0.93\n", "utf-8")
    nested_artifact = runner.store.save_artifact(
        path=str(nested),
        filename="display-name.csv",
        content_type="text/csv",
        size_bytes=nested.stat().st_size,
        checksum=hashlib.sha256(nested.read_bytes()).hexdigest(),
        frame_id=frame_id,
        root_frame_id=frame_id,
        project_id="project-domain",
    )
    try:
        started = runner.start_kernel(frame_id, "project-domain")
        assert started["state"] == "running"
        generation = runner.store.get_kernel_generation(
            started["generation_id"]
        )
        assert generation["bootstrap"]["version"] == 1
        assert generation["bootstrap"]["runtime_version"] == platform.python_version()

        code, checkpoint = _call(
            handler,
            "POST",
            f"/frames/{frame_id}/checkpoints",
            body={"reason": "recovery-test"},
        )
        assert code == 200
        assert checkpoint["generation_refs"]["python"]["bootstrap"]["version"] == 1
        assert checkpoint["recovery_recipe"]["artifact_hashes"] == {
            "results/out.csv": nested_artifact["checksum"]
        }

        runner.stop_kernel(frame_id, "project-domain")
        code, actions = _call(
            handler, "GET", f"/frames/{frame_id}/recovery/actions"
        )
        assert code == 200
        restore = next(item for item in actions["actions"] if item["id"] == "restore")
        assert restore["enabled"] is True

        code, restored = _call(
            handler,
            "POST",
            f"/frames/{frame_id}/recovery/actions/restore",
        )
        assert code == 200
        assert restored["ok"] is True
        assert restored["status"] == "active"
        assert restored["owner"]["kind"] == "recovery"
        state = runner._sessions[frame_id]
        assert state.kernels.alive("python")
        latest = runner.store.latest_kernel_generation(frame_id, "python")
        assert latest["recovered_from_generation_id"] == started["generation_id"]
        assert latest["bootstrap"]["version"] == 1
        assert nested.read_text("utf-8") == "score\n0.93\n"
        assert runner.session_domain.recovery_status(frame_id)["state"] == "active"
        assert any(event.get("type") == "recovery_state" for event in runner.hub.events)
    finally:
        runner.close()


def test_failed_fresh_recovery_keeps_exact_current_generation(
    monkeypatch, tmp_path
):
    runner, handler, frame_id = _setup(tmp_path)
    try:
        runner.start_kernel(frame_id, "project-domain")
        state = runner._sessions[frame_id]
        before = state.kernels.lease("python")

        with pytest.raises(gateway_mod.GatewayError) as confirmation:
            _call(
                handler,
                "POST",
                f"/frames/{frame_id}/recovery/actions/restart_fresh",
            )
        assert confirmation.value.code == 409
        assert "confirmation" in confirmation.value.message

        def fail_bootstrap(_runtime, _candidate, _manifest):
            raise RuntimeError("candidate dependency missing")

        monkeypatch.setattr(
            gateway_mod.SessionRecoveryRuntime,
            "_bootstrap_candidate",
            fail_bootstrap,
        )
        code, failed = _call(
            handler,
            "POST",
            f"/frames/{frame_id}/recovery/actions/restart_fresh",
            body={"confirm": True},
        )

        assert code == 409
        assert failed["status"] == "failed"
        current = state.kernels.lease("python")
        assert current == before
        assert current.kernel.is_alive()
        assert runner.store.latest_kernel_generation(
            frame_id, "python"
        )["generation_id"] == before.generation_id
        assert runner.session_domain.recovery_status(frame_id)["state"] == "failed"
    finally:
        runner.close()
