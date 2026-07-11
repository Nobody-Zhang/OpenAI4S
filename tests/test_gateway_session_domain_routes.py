from __future__ import annotations

import hashlib

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
