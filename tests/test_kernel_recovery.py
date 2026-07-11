from __future__ import annotations

import sqlite3
import threading

import pytest

from openai4s.kernel.recovery import (
    REPLAY_NEVER,
    REPLAY_SAFE,
    BootstrapManifest,
    KernelRecoveryOrchestrator,
    RecoveryRecipe,
    RecoveryStep,
    SidecarManifest,
    replay_safety_error,
)
from openai4s.storage.recovery import RecoveryJournalRepository


class _Candidate:
    def __init__(self, generation_id="candidate-1") -> None:
        self.generation_id = generation_id
        self.shutdown_calls = 0
        self.symbols = {"python": {"data", "model"}}
        self.artifacts = {"prediction.csv": "hash-prediction"}
        self.environment = {
            "interpreter": "/env/bin/python",
            "python_version": "3.12",
            "sdk_version": "sdk-1",
            "provenance_version": "prov-1",
        }

    def shutdown(self):
        self.shutdown_calls += 1


def _manifest():
    sidecar = SidecarManifest(
        "stats",
        b"def mean(values):\n    return sum(values) / len(values)\n",
        order=0,
        exports=("mean",),
        source_path="/snapshot/stats/kernel.py",
    )
    return BootstrapManifest(
        language="python",
        interpreter="/env/bin/python",
        runtime_version="3.12",
        working_directory="/workspace",
        environment={"name": "science", "hash": "env-hash"},
        sdk_version="sdk-1",
        provenance_version="prov-1",
        sidecars=(sidecar,),
    )


def _orchestrator(candidate, events, published, executed, *, bootstrap=None):
    return KernelRecoveryOrchestrator(
        build_candidate=lambda manifest: candidate,
        bootstrap_candidate=bootstrap
        or (lambda current, manifest: events.append("bootstrap")),
        hydrate_workspace=lambda current, payload: events.append(
            ("workspace", dict(payload))
        ),
        hydrate_artifact=lambda current, payload: events.append(
            ("artifact", dict(payload))
        ),
        execute_cell=lambda current, code, language: executed.append(
            (language, code)
        )
        or {"error": None},
        inspect_symbols=lambda current, language: current.symbols.get(language, set()),
        artifact_digest=lambda current, name: current.artifacts.get(name),
        inspect_environment=lambda current: current.environment,
        publish=lambda current: published.append(current.generation_id),
        journal=lambda event: events.append(("journal", event["phase"], event["status"])),
    )


def test_bootstrap_manifest_snapshots_exact_sidecar_bytes_and_detects_tampering():
    manifest = _manifest()
    record = manifest.record()

    restored = BootstrapManifest.from_record(record)
    assert restored.manifest_id == manifest.manifest_id
    assert restored.sidecars[0].source == manifest.sidecars[0].source
    assert restored.sidecars[0].sha256 == record["sidecars"][0]["sha256"]
    assert record["sidecars"][0]["source_path"] == "/snapshot/stats/kernel.py"

    record["sidecars"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        BootstrapManifest.from_record(record)


@pytest.mark.parametrize(
    ("code", "methods", "expected"),
    [
        ("host.submit_output({'ok': True}, ['Done'])", (), "submit_output"),
        ("host.bash('echo unsafe')", (), "bash"),
        ("value = host.unknown_service()", (), "unknown Host method"),
        ("import subprocess\nsubprocess.run(['true'])", (), "process"),
        ("from pathlib import Path\nPath('x').write_text('x')", (), "external state"),
        ("open('x', 'w').write('x')", (), "external state"),
        ("x = 1", ("write_file",), "unsafe Host methods"),
    ],
)
def test_replay_safety_fails_closed_for_external_side_effects(code, methods, expected):
    error = replay_safety_error(
        code, language="python", declared_host_methods=methods
    )
    assert expected in error


def test_replay_safety_allows_pure_computation_and_declared_read_only_host_calls():
    assert replay_safety_error("scores = [x*x for x in data]", language="python") is None
    assert (
        replay_safety_error(
            "rows = host.query({'sql': 'select 1'})",
            language="python",
            declared_host_methods=("query",),
        )
        is None
    )


def test_verified_candidate_is_published_only_after_hydration_replay_and_validation():
    candidate = _Candidate()
    events = []
    published = []
    executed = []
    orchestrator = _orchestrator(candidate, events, published, executed)
    recipe = RecoveryRecipe(
        steps=(
            RecoveryStep(
                "hydrate_workspace", {"tree_id": "tree-1"}, REPLAY_NEVER
            ),
            RecoveryStep(
                "hydrate_artifact",
                {"version_id": "version-1"},
                REPLAY_NEVER,
            ),
            RecoveryStep(
                "replay_cell",
                {"language": "python", "code": "scores = [x*x for x in data]"},
                REPLAY_SAFE,
                step_id="safe-cell",
            ),
        ),
        required_symbols={"python": ("data", "model")},
        artifact_hashes={"prediction.csv": "hash-prediction"},
        environment_requirements={"python_version": "3.12"},
    )

    result = orchestrator.restore(
        root_frame_id="root",
        branch_id="branch",
        manifest=_manifest(),
        recipe=recipe,
        source_generation_id="old-generation",
    )

    assert result.status == "active"
    assert result.replayed_steps == ("safe-cell",)
    assert result.issues == ()
    assert published == ["candidate-1"]
    assert executed == [("python", "scores = [x*x for x in data]")]
    assert candidate.shutdown_calls == 0
    assert ("journal", "publish", "completed") in events


def test_non_replayable_step_yields_partial_and_preserves_old_generation():
    candidate = _Candidate()
    events = []
    published = []
    executed = []
    old = {"generation_id": "old", "alive": True}
    orchestrator = _orchestrator(candidate, events, published, executed)
    recipe = RecoveryRecipe(
        steps=(
            RecoveryStep(
                "replay_cell",
                {"language": "python", "code": "host.bash('train.sh')"},
                REPLAY_SAFE,
                step_id="unsafe-cell",
            ),
        )
    )

    result = orchestrator.restore(
        root_frame_id="root",
        branch_id=None,
        manifest=_manifest(),
        recipe=recipe,
        source_generation_id=old["generation_id"],
    )

    assert result.status == "partial"
    assert result.skipped_steps == ("unsafe-cell",)
    assert result.issues[0]["type"] == "non_replayable"
    assert published == []
    assert executed == []
    assert candidate.shutdown_calls == 1
    assert old["alive"] is True


def test_prior_cells_without_namespace_coverage_cannot_be_declared_active():
    candidate = _Candidate()
    published = []
    result = _orchestrator(candidate, [], published, []).restore(
        root_frame_id="root",
        branch_id=None,
        manifest=_manifest(),
        recipe=RecoveryRecipe(namespace_coverage="unverified"),
        source_generation_id="old",
    )

    assert result.status == "partial"
    assert result.issues[0]["type"] == "namespace_unverified"
    assert published == []
    assert candidate.shutdown_calls == 1


def test_validation_failure_is_partial_and_bootstrap_failure_is_failed():
    candidate = _Candidate()
    candidate.symbols["python"].remove("model")
    partial = _orchestrator(candidate, [], [], []).restore(
        root_frame_id="root",
        branch_id=None,
        manifest=_manifest(),
        recipe=RecoveryRecipe(required_symbols={"python": ("model",)}),
        source_generation_id="old",
    )
    assert partial.status == "partial"
    assert partial.issues[0]["type"] == "missing_symbols"
    assert candidate.shutdown_calls == 1

    broken = _Candidate("candidate-broken")
    published = []
    failed = _orchestrator(
        broken,
        [],
        published,
        [],
        bootstrap=lambda current, manifest: (_ for _ in ()).throw(
            RuntimeError("missing package")
        ),
    ).restore(
        root_frame_id="root",
        branch_id=None,
        manifest=_manifest(),
        recipe=RecoveryRecipe(),
        source_generation_id="old",
    )
    assert failed.status == "failed"
    assert "missing package" in failed.issues[0]["error"]
    assert published == []
    assert broken.shutdown_calls == 1


def test_recovery_journal_is_append_only_and_survives_repository_reopen(tmp_path):
    connection = sqlite3.connect(tmp_path / "recovery.sqlite")
    connection.row_factory = sqlite3.Row
    lock = threading.RLock()
    repository = RecoveryJournalRepository(connection, lock, clock_ms=lambda: 1000)
    first = repository.append(
        recovery_id="recovery-1",
        root_frame_id="root",
        branch_id="branch",
        phase="build",
        status="completed",
        detail={"candidate": "gen-2"},
    )
    second = repository.append(
        recovery_id="recovery-1",
        root_frame_id="root",
        branch_id="branch",
        phase="validate",
        status="partial",
        detail={"missing": ["model"]},
    )

    reopened = RecoveryJournalRepository(connection, lock, clock_ms=lambda: 2000)
    rows = reopened.list(recovery_id="recovery-1")
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["entry_id"] == first["entry_id"]
    assert rows[1]["entry_id"] == second["entry_id"]
    assert rows[1]["detail"] == {"missing": ["model"]}
    connection.close()
