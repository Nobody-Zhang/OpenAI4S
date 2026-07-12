from __future__ import annotations

import base64
import hashlib
import importlib
import sqlite3
import sys
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
    frozen_sidecar_bootstrap_code,
    merge_bootstrap_sidecar_loads,
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
        execute_cell=lambda current, code, language: executed.append((language, code))
        or {"error": None},
        inspect_symbols=lambda current, language: current.symbols.get(language, set()),
        artifact_digest=lambda current, name: current.artifacts.get(name),
        inspect_environment=lambda current: current.environment,
        publish=lambda current: published.append(current.generation_id),
        journal=lambda event: events.append(
            ("journal", event["phase"], event["status"])
        ),
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


def test_bootstrap_manifest_v2_binds_worker_packages_locale_and_protocol_versions():
    base = BootstrapManifest(
        language="python",
        interpreter="/env/bin/python",
        runtime_version="unknown",
        working_directory="/workspace",
        environment={"environment_name": "science", "environment_root": "/env"},
    )
    observed = {
        "interpreter": "/env/bin/python",
        "runtime_version": "3.12.4",
        "prefix": "/env",
        "base_prefix": "/base",
        "sdk_version": "sdk-2",
        "provenance_version": "prov-2",
        "host_capability_version": "host-2",
        "package_manifest": [
            {"name": "zeta", "version": "2"},
            {"name": "Alpha", "version": "1"},
            {"name": "ALPHA", "version": "ignored-duplicate"},
        ],
        "locale": {"preferred_encoding": "UTF-8", "lc_ctype": "C.UTF-8"},
    }

    bound = base.with_observed_environment(observed)
    record = bound.record()
    restored = BootstrapManifest.from_record(record)

    assert record["version"] == 2
    assert bound.package_manifest == (("Alpha", "1"), ("zeta", "2"))
    assert bound.environment["interpreter_prefix"] == "/env"
    assert bound.environment["base_prefix"] == "/base"
    assert bound.host_capability_version == "host-2"
    assert len(bound.environment_hash or "") == 64
    assert restored == bound
    assert (
        base.with_observed_environment(
            {**observed, "package_manifest": [{"name": "Alpha", "version": "9"}]}
        ).environment_hash
        != bound.environment_hash
    )

    legacy = {
        key: value
        for key, value in base.record().items()
        if key
        not in {
            "host_capability_version",
            "package_manifest",
            "locale",
            "environment_hash",
        }
    }
    legacy["version"] = 1
    parsed_legacy = BootstrapManifest.from_record(legacy)
    assert parsed_legacy.version == 1
    assert parsed_legacy.record() == legacy


def _sidecar_event(module: str, source: bytes, order: int) -> dict:
    return {
        "event": "sidecar_loaded",
        "skill_name": module.partition(".")[0],
        "module": module,
        "source_b64": base64.b64encode(source).decode("ascii"),
        "sha256": hashlib.sha256(source).hexdigest(),
        "expected_sha256": hashlib.sha256(source).hexdigest(),
        "source_path": f"/mutable/{module.partition('.')[0]}/kernel.py",
        "order": order,
        "import_mode": "module",
    }


def test_runtime_sidecar_events_extend_manifest_in_exact_load_order():
    bootstrap = BootstrapManifest(
        language="python",
        interpreter="/env/bin/python",
        runtime_version="3.12",
        working_directory="/workspace",
    ).record()
    first = _sidecar_event("alpha.kernel", b"VALUE = 'alpha'\n", 0)
    second = _sidecar_event("beta.kernel", b"VALUE = 'beta'\n", 1)

    after_first = merge_bootstrap_sidecar_loads(bootstrap, [first])
    complete = merge_bootstrap_sidecar_loads(after_first, [second])
    restored = BootstrapManifest.from_record(complete)

    assert [item.name for item in restored.sidecars] == [
        "alpha.kernel",
        "beta.kernel",
    ]
    assert [item.order for item in restored.sidecars] == [0, 1]
    assert [item["module"] for item in complete["loaded_sidecars"]] == [
        "alpha.kernel",
        "beta.kernel",
    ]
    # Processing an already-committed exact event is idempotent.
    assert merge_bootstrap_sidecar_loads(complete, [first]) == complete


def test_frozen_sidecar_recovery_uses_manifest_bytes_not_changed_disk(tmp_path):
    skill = tmp_path / "frozen_skill"
    skill.mkdir()
    path = skill / "kernel.py"
    old_source = b"VALUE = 'frozen-old'\n"
    path.write_bytes(old_source)
    sidecar = SidecarManifest(
        name="frozen_skill.kernel",
        source=old_source,
        order=0,
        source_path=str(path),
    )
    path.write_text("VALUE = 'mutable-new'\n", encoding="utf-8")

    try:
        exec(frozen_sidecar_bootstrap_code(sidecar), {})
        module = importlib.import_module("frozen_skill.kernel")
        assert module.VALUE == "frozen-old"
        assert module.__openai4s_frozen_sidecar_sha256__ == sidecar.sha256
    finally:
        sys.modules.pop("frozen_skill.kernel", None)
        sys.modules.pop("frozen_skill", None)


@pytest.mark.parametrize(
    "source",
    [
        b"from .helper import VALUE\n",
        b"import local_skill.helper\n",
        b"from local_skill import helper\n",
    ],
)
def test_sidecar_manifest_rejects_unfrozen_local_imports(source):
    with pytest.raises(ValueError, match="unfrozen local import"):
        SidecarManifest(
            name="local_skill.kernel",
            source=source,
            order=0,
            source_path="/mutable/local_skill/kernel.py",
        )


def test_frozen_sidecar_package_path_cannot_load_changed_sibling(tmp_path, monkeypatch):
    skill = tmp_path / "frozen_dynamic_skill"
    skill.mkdir()
    (skill / "__init__.py").write_text("", encoding="utf-8")
    helper = skill / "helper.py"
    helper.write_text("VALUE = 'helper-old'\n", encoding="utf-8")
    source = (
        b"import importlib\n"
        b"VALUE = importlib.import_module(__package__ + '.helper').VALUE\n"
    )
    sidecar = SidecarManifest(
        name="frozen_dynamic_skill.kernel",
        source=source,
        order=0,
        source_path=str(skill / "kernel.py"),
    )
    helper.write_text("VALUE = 'helper-new'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with pytest.raises(ModuleNotFoundError, match="helper"):
            exec(frozen_sidecar_bootstrap_code(sidecar), {})
        assert "frozen_dynamic_skill.helper" not in sys.modules
    finally:
        sys.modules.pop("frozen_dynamic_skill.helper", None)
        sys.modules.pop("frozen_dynamic_skill.kernel", None)
        sys.modules.pop("frozen_dynamic_skill", None)


def test_sidecar_event_tampering_and_capture_failure_fail_closed():
    bootstrap = BootstrapManifest(
        language="python",
        interpreter="/env/bin/python",
        runtime_version="3.12",
        working_directory="/workspace",
    ).record()
    event = _sidecar_event("stats.kernel", b"VALUE = 1\n", 0)
    event["source_b64"] = base64.b64encode(b"VALUE = 2\n").decode("ascii")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_bootstrap_sidecar_loads(bootstrap, [event])

    wrong_bootstrap_hash = _sidecar_event("stats.kernel", b"VALUE = 1\n", 0)
    wrong_bootstrap_hash["expected_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bootstrap hash"):
        merge_bootstrap_sidecar_loads(bootstrap, [wrong_bootstrap_hash])

    non_integer_order = _sidecar_event("stats.kernel", b"VALUE = 1\n", 0)
    non_integer_order["order"] = 0.0
    with pytest.raises(ValueError, match="order must be an integer"):
        merge_bootstrap_sidecar_loads(bootstrap, [non_integer_order])

    failed = dict(bootstrap)
    failed["sidecar_capture_status"] = "failed"
    failed["sidecar_capture_error"] = "worker event was invalid"
    with pytest.raises(ValueError, match="capture is incomplete"):
        BootstrapManifest.from_record(failed)


@pytest.mark.parametrize(
    ("code", "methods", "expected"),
    [
        ("host.submit_output({'ok': True}, ['Done'])", (), "submit_output"),
        ("host.bash('echo unsafe')", (), "bash"),
        ("value = host.unknown_service()", (), "unknown Host method"),
        ("import subprocess\nsubprocess.run(['true'])", (), "process"),
        # ``from <module> import ...`` is caught by the import blocklist (the
        # module, not the imported symbol name); previously it slipped past to
        # the write-method attribute check.
        ("from pathlib import Path\nPath('x').write_text('x')", (), "imports a direct"),
        ("from shutil import rmtree\nrmtree('x')", (), "imports a direct"),
        ("scores.to_csv('x')", (), "external state"),
        ("open('x', 'w').write('x')", (), "external state"),
        ("x = 1", ("write_file",), "unsafe Host methods"),
    ],
)
def test_replay_safety_fails_closed_for_external_side_effects(code, methods, expected):
    error = replay_safety_error(code, language="python", declared_host_methods=methods)
    assert expected in error


def test_replay_safety_allows_pure_computation_and_declared_read_only_host_calls():
    assert (
        replay_safety_error("scores = [x*x for x in data]", language="python") is None
    )
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
            RecoveryStep("hydrate_workspace", {"tree_id": "tree-1"}, REPLAY_NEVER),
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


def test_replay_step_source_hash_is_rechecked_before_execution():
    candidate = _Candidate()
    executed = []
    published = []
    code = "scores = [x*x for x in data]"
    recipe = RecoveryRecipe(
        steps=(
            RecoveryStep(
                "replay_cell",
                {
                    "language": "python",
                    "code": code,
                    "code_hash": hashlib.sha256(b"different source").hexdigest(),
                },
                REPLAY_SAFE,
                step_id="tampered-cell",
            ),
        )
    )

    result = _orchestrator(candidate, [], published, executed).restore(
        root_frame_id="root",
        branch_id=None,
        manifest=_manifest(),
        recipe=recipe,
        source_generation_id="old",
    )

    assert result.status == "partial"
    assert result.skipped_steps == ("tampered-cell",)
    assert "source hash" in result.issues[0]["reason"]
    assert executed == []
    assert published == []


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
