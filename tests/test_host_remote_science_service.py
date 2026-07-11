"""Direct contracts for real remote folding and mutation scoring."""

from __future__ import annotations

import base64
import json
import subprocess
from types import SimpleNamespace

import pytest

from openai4s.host.remote_science import RemoteScienceService


class FakeRegistry:
    def __init__(self, capabilities=None) -> None:
        self.capabilities = capabilities or {}
        self.calls: list[str] = []

    def capability_host(self, capability: str):
        self.calls.append(capability)
        return self.capabilities.get(capability, (None, None))


class FakeRunner:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def process(stdout: str, stderr: str = "", returncode: int = 0):
    return SimpleNamespace(
        stdout=stdout.encode(),
        stderr=stderr.encode(),
        returncode=returncode,
    )


def encoded(value: str | dict) -> str:
    raw = json.dumps(value) if isinstance(value, dict) else value
    return base64.b64encode(raw.encode()).decode()


def fold_output(*, provenance: str | None = None, manifest: str | None = None):
    manifest = manifest or json.dumps(
        {
            "engine": "protenix-test",
            "mean_plddt": 88.5,
            "ptm": 0.71,
            "length": 4,
            "residues_modeled": 4,
            "msa": True,
        }
    )
    provenance_block = (
        "===PROVENANCE_JSON===\n"
        f"{provenance}\n"
        "===END_PROVENANCE_JSON===\n"
        if provenance is not None
        else ""
    )
    return (
        "===FOLD_RESULT_JSON===\n"
        f"{manifest}\n"
        "===END_FOLD_RESULT_JSON===\n"
        "===FOLD_PDB_B64===\n"
        f"{encoded('ATOM\n')}\n"
        "===FOLD_PLDDT_CSV_B64===\n"
        f"{encoded('residue,plddt\n1,90\n')}\n"
        "===FOLD_CONFIDENCE_JSON_B64===\n"
        f"{encoded({'overall': 0.9})}\n"
        f"{provenance_block}"
        "===FOLD_DONE===\n"
    )


def mutation_output(*, provenance: str | None = None, summary: str | None = None):
    summary = summary or json.dumps(
        {
            "mean_score": -0.2,
            "top5": [{"mutation": "A1C", "score": 1.2}],
            "length": 3,
        }
    )
    provenance_block = (
        "===PROVENANCE_JSON===\n"
        f"{provenance}\n"
        "===END_PROVENANCE_JSON===\n"
        if provenance is not None
        else ""
    )
    return (
        "===MUT_RESULT_JSON===\n"
        f"{summary}\n"
        "===END_MUT_RESULT_JSON===\n"
        "===MUT_CSV_B64===\n"
        f"{encoded('mutation,score\nA1C,1.2\n')}\n"
        f"{provenance_block}"
        "===MUT_DONE===\n"
    )


def test_provenance_buffer_parses_best_effort_and_drains_by_identity():
    service = RemoteScienceService()
    del service._remote_provenance
    assert service.pop_remote_provenance() == []
    service.record_remote_provenance(
        "fold", "gpu-a", "protenix", "/jobs/a", ' {"cuda":"12"} '
    )
    service.record_remote_provenance(
        "score_mutations", "gpu-b", None, "/jobs/b", "not-json"
    )
    service.record_remote_provenance("fold", "gpu-c", None, "/jobs/c", None)

    buffered = service.pop_remote_provenance()
    assert buffered == [
        {
            "service": "fold",
            "host": "gpu-a",
            "engine": "protenix",
            "remote_dir": "/jobs/a",
            "env": {"cuda": "12"},
        },
        {
            "service": "score_mutations",
            "host": "gpu-b",
            "engine": None,
            "remote_dir": "/jobs/b",
            "env": None,
        },
        {
            "service": "fold",
            "host": "gpu-c",
            "engine": None,
            "remote_dir": "/jobs/c",
            "env": None,
        },
    ]
    buffered.append({"external": True})
    assert service.pop_remote_provenance() == []

    delegated = []
    service = RemoteScienceService(
        provenance_recorder=lambda *args: delegated.append(args)
    )
    service._record_provenance("fold", "gpu", "engine", "/job", "{}")
    assert delegated == [("fold", "gpu", "engine", "/job", "{}")]
    assert service.pop_remote_provenance() == []


def test_fold_validation_and_no_fabrication_errors_are_exact():
    registry = FakeRegistry()
    service = RemoteScienceService(registry_factory=lambda: registry)

    assert service.fold({}) == {
        "error": "fold: a protein 'sequence' (amino acids) is required"
    }
    assert service.fold({"sequence": "123 BZX"}) == {
        "error": "fold: a protein 'sequence' (amino acids) is required"
    }
    assert service.fold({"sequence": "A" * 1201}) == {
        "error": "fold: sequence too long (1201 aa); the demo host caps "
        "single-sequence folds at 1200 aa"
    }
    assert registry.calls == []
    assert service.fold({"sequence": "ACDE"}) == {
        "error": "fold: no remote GPU host with a folding service is configured "
        "(Settings → Remote GPU). Refusing to fabricate a structure — configure "
        "a host first."
    }
    assert registry.calls == ["fold"]


def test_fold_preserves_ssh_argv_markers_result_and_environment_provenance():
    registry = FakeRegistry({"fold": ("gpu-a", {"engine": "registered-engine"})})
    runner = FakeRunner(
        process(
            fold_output(provenance='{"cuda":"12.4","weights":"sha256:abc"}'),
            returncode=17,
        )
    )
    environment = {
        "OPENAI4S_FOLD_SCRIPT": "/opt/fold script.sh",
        "OPENAI4S_FOLD_JOBS_DIR": "/jobs base",
    }
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=runner,
        environment=lambda: environment,
        job_suffix=lambda: "abcd1234",
    )

    result = service.fold(
        {
            "sequence": " ac dxE ",
            "name": "My Protein!*",
            "gpu": "2",
            "cycle": "3",
            "step": "4",
        }
    )

    assert runner.calls == [
        (
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "BatchMode=yes",
                "gpu-a",
                "mkdir -p '/jobs base/MyProtein_abcd1234' && "
                "'/opt/fold script.sh' --seq ACDE --name MyProtein --out "
                "'/jobs base/MyProtein_abcd1234' --gpu 2 --cycle 3 --step 4",
            ],
            {"capture_output": True, "timeout": 900},
        )
    ]
    assert result == {
        "ok": True,
        "pdb": "ATOM\n",
        "plddt_csv": "residue,plddt\n1,90\n",
        "confidence": {"overall": 0.9},
        "mean_plddt": 88.5,
        "ptm": 0.71,
        "length": 4,
        "residues_modeled": 4,
        "engine": "protenix-test",
        "msa": True,
        "host": "gpu-a (8×NVIDIA A100-80GB · Protenix AF3-class)",
        "remote_dir": "/jobs base/MyProtein_abcd1234",
    }
    assert service.pop_remote_provenance() == [
        {
            "service": "fold",
            "host": "gpu-a",
            "engine": "protenix-test",
            "remote_dir": "/jobs base/MyProtein_abcd1234",
            "env": {"cuda": "12.4", "weights": "sha256:abc"},
        }
    ]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            subprocess.TimeoutExpired("ssh", 900),
            "fold: timed out after 900s on gpu-a",
        ),
        (OSError("offline"), "fold: ssh to gpu-a failed: offline"),
    ],
)
def test_fold_transport_failures_are_soft_errors(failure, expected):
    registry = FakeRegistry({"fold": ("gpu-a", {"script": "/fold"})})
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=FakeRunner(failure),
        job_suffix=lambda: "job",
    )
    assert service.fold({"sequence": "ACDE"}) == {"error": expected}
    assert service.pop_remote_provenance() == []


def test_fold_incomplete_and_parse_failures_keep_exact_diagnostics():
    registry = FakeRegistry({"fold": ("gpu-a", {"script": "/fold"})})
    runner = FakeRunner(process("stdout tail", "  remote failed\n", returncode=9))
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=runner,
        job_suffix=lambda: "job",
    )
    assert service.fold({"sequence": "ACDE"}) == {
        "error": "fold: prediction did not complete on gpu-a (rc=9). tail: "
        "remote failed"
    }

    runner.result = process(fold_output(manifest="not-json"))
    result = service.fold({"sequence": "ACDE"})
    assert result["error"].startswith("fold: could not parse prediction output: ")
    assert service.pop_remote_provenance() == []


def test_mutation_validation_and_no_fabrication_errors_are_exact():
    registry = FakeRegistry()
    service = RemoteScienceService(registry_factory=lambda: registry)
    assert service.score_mutations({}) == {
        "error": "score_mutations: a protein 'sequence' is required"
    }
    assert service.score_mutations({"sequence": "A" * 1025}) == {
        "error": "score_mutations: sequence too long (1025 aa); cap is 1024"
    }
    assert registry.calls == []
    assert service.score_mutations({"sequence": "ACD"}) == {
        "error": "score_mutations: no remote GPU host has a mutation-scoring "
        "service configured, so there is no real predictor available. Do NOT "
        "fabricate scores (no np.random, no BLOSUM-as-ESM, no fake heatmap) — "
        "report that this step cannot be done for real. Provision a service via "
        "Settings → Remote GPU."
    }
    registry.capabilities["score_mutations"] = ("gpu-b", {"engine": "esm"})
    assert service.score_mutations({"sequence": "ACD"}) == {
        "error": "score_mutations: host gpu-b has no script recorded"
    }


def test_mutation_scoring_preserves_ssh_result_and_provenance_contract():
    registry = FakeRegistry(
        {
            "score_mutations": (
                "gpu-b",
                {"script": "/opt/esm score.sh", "engine": "ESM-2"},
            )
        }
    )
    runner = FakeRunner(
        process(mutation_output(provenance='{"torch":"2.5"}'), returncode=5)
    )
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=runner,
        environment=lambda: {"OPENAI4S_ESM_JOBS_DIR": "/esm jobs"},
        job_suffix=lambda: "ef567890",
    )

    result = service.score_mutations(
        {
            "sequence": "acdx",
            "name": "Variant Set!",
            "gpu": "3",
            "positions": [1, "2"],
        }
    )

    assert runner.calls == [
        (
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "BatchMode=yes",
                "gpu-b",
                "mkdir -p '/esm jobs/VariantSet_ef567890' && "
                "'/opt/esm score.sh' --seq ACD --name VariantSet --out "
                "'/esm jobs/VariantSet_ef567890' --gpu 3 --positions 1,2",
            ],
            {"capture_output": True, "timeout": 1200},
        )
    ]
    assert result == {
        "ok": True,
        "scores_csv": "mutation,score\nA1C,1.2\n",
        "summary": {
            "mean_score": -0.2,
            "top5": [{"mutation": "A1C", "score": 1.2}],
            "length": 3,
        },
        "mean_score": -0.2,
        "top5": [{"mutation": "A1C", "score": 1.2}],
        "length": 3,
        "model": "ESM-2",
        "host": "gpu-b · ESM-2",
        "remote_dir": "/esm jobs/VariantSet_ef567890",
    }
    assert service.pop_remote_provenance() == [
        {
            "service": "score_mutations",
            "host": "gpu-b",
            "engine": "ESM-2",
            "remote_dir": "/esm jobs/VariantSet_ef567890",
            "env": {"torch": "2.5"},
        }
    ]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            subprocess.TimeoutExpired("ssh", 1200),
            "score_mutations: timed out after 1200s on gpu-b",
        ),
        (OSError("offline"), "score_mutations: ssh to gpu-b failed: offline"),
    ],
)
def test_mutation_transport_failures_are_soft_errors(failure, expected):
    registry = FakeRegistry(
        {"score_mutations": ("gpu-b", {"script": "/score", "engine": "ESM"})}
    )
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=FakeRunner(failure),
        job_suffix=lambda: "job",
    )
    assert service.score_mutations({"sequence": "ACD"}) == {"error": expected}
    assert service.pop_remote_provenance() == []


def test_mutation_incomplete_and_parse_failures_keep_exact_diagnostics():
    registry = FakeRegistry(
        {"score_mutations": ("gpu-b", {"script": "/score", "engine": "ESM"})}
    )
    runner = FakeRunner(process("stdout tail", " remote failed ", returncode=11))
    service = RemoteScienceService(
        registry_factory=lambda: registry,
        run_command=runner,
        job_suffix=lambda: "job",
    )
    assert service.score_mutations({"sequence": "ACD"}) == {
        "error": "score_mutations: no real result from gpu-b (rc=11) — report "
        "the failure, do NOT fabricate. tail: remote failed"
    }

    runner.result = process(mutation_output(summary="not-json"))
    result = service.score_mutations({"sequence": "ACD"})
    assert result["error"].startswith("score_mutations: could not parse output: ")
    assert service.pop_remote_provenance() == []
