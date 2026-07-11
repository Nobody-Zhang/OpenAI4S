"""Direct contracts for verified remote science capabilities."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from openai4s.host.remote_capabilities import (
    RemoteCapabilityService,
    normalize_remote_capability_probe,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.hosts = {}
        self.default = None
        self.saved = []

    def list_hosts(self):
        return self.hosts

    def default_host(self):
        return self.default

    def get_host(self, alias):
        return self.hosts.get(alias)

    def set_capability(self, alias, capability, metadata):
        self.saved.append((alias, capability, dict(metadata)))
        saved = {**metadata, "verified_at": 1234}
        self.hosts[alias].setdefault("capabilities", {})[capability] = saved


def _service(registry, runner):
    return RemoteCapabilityService(
        registry_factory=lambda: registry,
        run_command=runner,
    )


def test_status_projects_only_registered_capabilities_and_missing_core():
    registry = FakeRegistry()
    registry.default = "gpu-a"
    registry.hosts = {
        "gpu-a": {
            "label": "Lab GPU",
            "gpus": "A100",
            "gpu_count": 2,
            "capabilities": {
                "fold": {
                    "engine": "protenix",
                    "script": "/srv/fold.sh",
                    "verified_at": 99,
                },
                "unverified": None,
            },
        }
    }
    service = _service(registry, lambda *_a, **_kw: None)

    assert service.status() == {
        "configured": True,
        "default_host": "gpu-a",
        "hosts": [
            {
                "alias": "gpu-a",
                "label": "Lab GPU",
                "provider": "ssh:gpu-a",
                "gpus": "A100",
                "gpu_count": 2,
                "capabilities": [
                    {
                        "name": "fold",
                        "engine": "protenix",
                        "script": "/srv/fold.sh",
                        "verified": True,
                        "verified_at": 99,
                    },
                    {
                        "name": "unverified",
                        "engine": None,
                        "script": None,
                        "verified": False,
                        "verified_at": None,
                    },
                ],
            }
        ],
        "core_capabilities": ["fold", "score_mutations"],
        "missing_core_capabilities": ["score_mutations"],
    }


def test_register_verifies_exact_ssh_command_before_registry_write():
    registry = FakeRegistry()
    registry.default = "gpu-a"
    registry.hosts["gpu-a"] = {"capabilities": {}}
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    service = _service(registry, runner)
    spec = {
        "alias": " gpu-a ",
        "cap": " fold ",
        "script": "/srv/Model Runner/fold.sh",
        "invoke": "{script} {input}",
        "engine": "protenix",
        "markers": {"done": "result.json"},
        "notes": "verified fixture",
    }
    result = service.register(spec)

    assert calls == [
        (
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "BatchMode=yes",
                "gpu-a",
                "test -e '/srv/Model Runner/fold.sh'",
            ],
            {"capture_output": True, "text": True, "timeout": 90},
        )
    ]
    assert registry.saved == [
        (
            "gpu-a",
            "fold",
            {
                "script": "/srv/Model Runner/fold.sh",
                "invoke": "{script} {input}",
                "engine": "protenix",
                "markers": {"done": "result.json"},
                "notes": "verified fixture",
                "probe": {
                    "kind": "path_exists",
                    "path": "/srv/Model Runner/fold.sh",
                },
                "verification": "test -e '/srv/Model Runner/fold.sh'",
            },
        )
    ]
    assert result["ok"] is True
    assert result["alias"] == "gpu-a"
    assert result["capability"] == "fold"
    assert result["status"]["missing_core_capabilities"] == ["score_mutations"]


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({}, "alias is required"),
        ({"alias": "gpu-a"}, "capability is required"),
        (
            {"alias": "missing", "capability": "fold", "script": "/x"},
            "unknown remote GPU host 'missing'",
        ),
        (
            {
                "alias": "gpu-a",
                "capability": "fold",
                "script": "/x; whoami",
            },
            "invalid probe",
        ),
    ],
)
def test_validation_soft_fails_before_starting_ssh(spec, expected):
    registry = FakeRegistry()
    registry.hosts["gpu-a"] = {"capabilities": {}}
    calls = []
    service = _service(registry, lambda *a, **kw: calls.append((a, kw)))

    result = service.register(spec)

    assert expected in result["error"]
    assert calls == []
    assert registry.saved == []


@pytest.mark.parametrize(
    "outcome, expected",
    [
        (
            subprocess.TimeoutExpired(["ssh"], 90),
            "verification timed out on gpu-a",
        ),
        (OSError("ssh missing"), "ssh to gpu-a failed: ssh missing"),
        (
            SimpleNamespace(returncode=7, stdout="fallback", stderr="denied"),
            "verification failed on gpu-a (rc=7). tail: denied",
        ),
    ],
)
def test_transport_failures_never_register(outcome, expected):
    registry = FakeRegistry()
    registry.hosts["gpu-a"] = {"capabilities": {}}

    def runner(*_args, **_kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    service = _service(registry, runner)
    result = service.register(
        {"alias": "gpu-a", "capability": "fold", "script": "/srv/fold.sh"}
    )

    assert expected in result["error"]
    assert registry.saved == []


def test_probe_normalizer_keeps_structured_and_legacy_grammar():
    assert normalize_remote_capability_probe(
        {"probe": {"kind": "executable_exists", "binary": "python3.11"}}
    ) == (
        {"kind": "executable_exists", "binary": "python3.11"},
        "which python3.11",
    )
    assert normalize_remote_capability_probe(
        {"verify_command": "test -e '/srv/Model Runner/fold.sh'"}
    ) == (
        {"kind": "path_exists", "path": "/srv/Model Runner/fold.sh"},
        "test -e '/srv/Model Runner/fold.sh'",
    )
    with pytest.raises(ValueError, match="forbidden shell syntax"):
        normalize_remote_capability_probe(
            {"probe": {"kind": "path_exists", "path": "/srv/x; whoami"}}
        )
