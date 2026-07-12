"""Restart-safe delegation persistence contracts."""

from __future__ import annotations

import threading
import time

import pytest

import openai4s.agent.loop as loop_mod
from openai4s.agent.delegation import (
    DelegationBudget,
    DelegationError,
    DelegationRunner,
)
from openai4s.agent.models import RunState
from openai4s.config import get_config
from openai4s.store import get_store


def _submitted(output=None):
    return {
        "stop_reason": "submitted",
        "submitted_output": {
            "output": output if output is not None else {"ok": True},
            "completion_bullets": ["child complete"],
        },
        "final_message": None,
    }


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition not reached before timeout")


def _root_store():
    cfg = get_config()
    store = get_store(cfg.db_path)
    root = store.new_frame(kind="turn", project_id="science")
    return cfg, store, root


def test_restart_reconstructs_terminal_children_and_budget(monkeypatch):
    monkeypatch.setattr(
        loop_mod.Agent,
        "run",
        lambda self, task: _submitted(
            {"task": task, "api_key": "ark-secret-value-123456"}
        ),
    )
    cfg, store, root = _root_store()
    first = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        budget=DelegationBudget(root, limit=2),
        owner_instance_id="daemon-a",
        runner_instance_id="runner-a",
    )
    initial = first({"request": "first durable child"})
    first.close()

    reopened = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        owner_instance_id="daemon-b",
        runner_instance_id="runner-b",
    )
    assert reopened.children()[0]["child_id"] == initial["child_id"]
    assert reopened.children()[0]["status"] == "done"
    assert reopened.delegation_stats()["spawned_session"] == 1
    assert "ark-secret-value" not in repr(store.delegation_tree(root))

    second = reopened({"request": "second durable child"})
    assert second["child_id"] != initial["child_id"]
    assert reopened.delegation_stats()["spawned_session"] == 2
    with pytest.raises(DelegationError, match="already spawned 2"):
        reopened({"request": "budget must not reset"})
    reopened.close()


def test_restart_stops_dead_live_child_and_fences_old_lease(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked_run(self, task):
        del self, task
        started.set()
        assert release.wait(2)
        return _submitted({"late": True})

    monkeypatch.setattr(loop_mod.Agent, "run", blocked_run)
    cfg, store, root = _root_store()
    first = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        owner_instance_id="daemon-dead",
        runner_instance_id="runner-dead",
    )
    handle = first({"request": "still running", "wait": False})
    assert started.wait(2)
    assert store.delegation_tree(root)["stats"]["running"] == 1

    reopened = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        owner_instance_id="daemon-new",
        runner_instance_id="runner-new",
    )
    restored = reopened.children()[0]
    assert restored["child_id"] == handle["child_id"]
    assert restored["status"] == "stopped"
    assert restored["output"] is None
    assert store.delegation_budget(root)["active"] == 0

    release.set()
    first.collect({"child_ids": [handle["child_id"]]})
    assert store.delegation_tree(root)["children"][0]["status"] == "stopped"
    first.close()
    reopened.close()


def test_steering_delivery_state_is_durable_and_text_safe(monkeypatch):
    first_boundary = threading.Event()
    continue_turn = threading.Event()
    observed: list[dict] = []

    def boundary_run(self, task):
        state = RunState(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": task},
            ],
            max_turns=self.max_turns,
        )
        self.context_policy.prepare(state)
        first_boundary.set()
        assert continue_turn.wait(2)
        state.turn = 1
        self.context_policy.prepare(state)
        observed.extend(state.messages)
        return _submitted()

    monkeypatch.setattr(loop_mod.Agent, "run", boundary_run)
    cfg, store, root = _root_store()
    runner = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        owner_instance_id="daemon-steer",
        runner_instance_id="runner-steer",
    )
    handle = runner({"request": "initial", "wait": False})
    assert first_boundary.wait(2)
    runner.send_message(
        {
            "child_id": handle["child_id"],
            "message": "Use dataset B; api_key=ark-secret-message-123456",
        }
    )
    queued = store.delegation_tree(root)["children"][0]["steering"]
    assert queued["queued"] == 1
    assert queued["messages"][0]["status"] == "queued"
    assert "dataset B" not in repr(queued)
    assert "ark-secret-message" not in repr(store.delegation_tree(root))

    continue_turn.set()
    runner.collect({"child_ids": [handle["child_id"]]})
    delivered = store.delegation_tree(root)["children"][0]["steering"]
    assert delivered["queued"] == 0
    assert delivered["delivered"] == 1
    assert delivered["messages"][0]["boundary"] == 2
    assert any("Use dataset B" in item.get("content", "") for item in observed)
    runner.close()


def test_parent_cancel_persists_every_descendant_stopped(monkeypatch):
    child_ready = threading.Event()
    grandchild_ready = threading.Event()

    def cancellable_run(self, task):
        if self.delegate_depth == 1:
            self.dispatcher._delegate_fn({"request": "nested", "wait": False})
            child_ready.set()
            assert grandchild_ready.wait(2)
        else:
            grandchild_ready.set()
        _wait_for(lambda: self.cancellation.cancelled())
        return {
            "stop_reason": "cancelled",
            "submitted_output": None,
            "final_message": None,
        }

    monkeypatch.setattr(loop_mod.Agent, "run", cancellable_run)
    cfg, store, root = _root_store()
    runner = DelegationRunner(
        cfg,
        parent_frame_id=root,
        store=store,
        owner_instance_id="daemon-cancel",
        runner_instance_id="runner-cancel",
    )
    parent = runner({"request": "parent", "wait": False})
    assert child_ready.wait(2)
    assert grandchild_ready.wait(2)
    runner.stop_child(parent["child_id"])
    runner.collect({"child_ids": [parent["child_id"]]})
    _wait_for(lambda: store.delegation_tree(root)["stats"]["stopped"] == 2)
    projection = store.delegation_tree(root)
    assert projection["stats"]["stopped"] == 2
    assert projection["budget"]["active"] == 0
    assert {item["parent_child_id"] for item in projection["children"]} == {
        None,
        parent["child_id"],
    }
    runner.close()


def test_session_deletion_removes_delegation_projection(monkeypatch):
    monkeypatch.setattr(loop_mod.Agent, "run", lambda self, task: _submitted())
    cfg, store, root = _root_store()
    runner = DelegationRunner(cfg, parent_frame_id=root, store=store)
    runner({"request": "durable child"})
    assert store.delegation_tree(root)["stats"]["total"] == 1

    runner.close()
    store.delete_frame(root)

    assert store.delegation_tree(root)["initialized"] is False
