"""Offline contract tests for the Python/R OS sandbox spawn boundary."""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import openai4s.security.sandbox as sandbox_module
from openai4s.kernel import Kernel
from openai4s.security.sandbox import (
    SandboxConfigurationError,
    SandboxStatus,
    SandboxUnavailableError,
    build_seatbelt_profile,
    create_kernel_sandbox,
    wrap_bwrap_command,
    wrap_seatbelt_command,
)


def _passing_runner(calls: list | None = None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append((list(command), kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"ok": true, "checks": {'
                '"network_blocked": true, "outside_write_blocked": true, '
                '"temp_write": true, "workspace_write": true}}\n'
            ),
            stderr="",
        )

    return run


def _failing_runner(command, **kwargs):
    del command, kwargs
    return SimpleNamespace(
        returncode=71,
        stdout="",
        stderr="sandbox_apply: Operation not permitted\n",
    )


def test_seatbelt_profile_escapes_paths_and_blocks_network_by_default():
    workspace = '/tmp/project with space/quote"\\tail) (allow network*)'
    temp_dir = "/tmp/private temp"

    profile = build_seatbelt_profile(workspace, temp_dir)

    assert "(deny network*)" in profile
    assert '(subpath "/tmp/project with space/quote\\"\\\\tail) ' in profile
    assert '(subpath "/tmp/private temp")' in profile
    assert profile.count("(allow default)") == 1
    # The path stays one quoted Scheme string; its quote cannot terminate the
    # path and inject the attacker-shaped policy text that follows it.
    assert 'quote"\\tail' not in profile


def test_seatbelt_raw_network_switch_is_explicit_and_argv_has_no_shell():
    command = ["/usr/bin/python3", "-c", "print('ok')"]
    wrapped = wrap_seatbelt_command(
        command,
        executable="/usr/bin/sandbox-exec",
        workspace="/tmp/work",
        temp_dir="/tmp/private",
        allow_raw_network=True,
    )

    assert wrapped[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "(deny network*)" not in wrapped[2]
    assert wrapped[3:] == command


def test_bwrap_mounts_only_workspace_and_private_temp_writable():
    workspace = "/tmp/project; still-one-argument"
    temp_dir = "/tmp/kernel-private"
    command = ["/usr/bin/python3", "-u", "/code/worker.py"]

    wrapped = wrap_bwrap_command(
        command,
        executable="/usr/bin/bwrap",
        workspace=workspace,
        temp_dir=temp_dir,
    )

    assert wrapped[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in wrapped
    assert wrapped[wrapped.index("--ro-bind") + 1 : wrapped.index("--ro-bind") + 3] == [
        "/",
        "/",
    ]
    bind_positions = [i for i, value in enumerate(wrapped) if value == "--bind"]
    assert [[wrapped[i + 1], wrapped[i + 2]] for i in bind_positions] == [
        [workspace, workspace],
        [temp_dir, temp_dir],
    ]
    assert wrapped[wrapped.index("--") + 1 :] == command


def test_bwrap_raw_network_compatibility_switch_only_removes_network_namespace():
    wrapped = wrap_bwrap_command(
        ["/bin/true"],
        executable="bwrap",
        workspace="/workspace",
        temp_dir="/kernel-tmp",
        allow_raw_network=True,
    )

    assert "--unshare-net" not in wrapped
    assert ["--ro-bind", "/", "/"] == wrapped[
        wrapped.index("--ro-bind") : wrapped.index("--ro-bind") + 3
    ]


def test_off_is_explicit_and_skips_detection_and_self_test(tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError((args, kwargs))

    sandbox = create_kernel_sandbox(
        tmp_path,
        mode="off",
        platform_name="darwin",
        which=unexpected,
        runner=unexpected,
    )

    assert sandbox.status.state == "disabled"
    assert sandbox.status.mode == "off"
    assert sandbox.status.enforced is False
    assert sandbox.status.network_policy == "not_enforced"
    assert sandbox.wrap_command(["python", "worker.py"]) == ["python", "worker.py"]


def test_auto_missing_backend_falls_back_with_visible_status_and_warning(tmp_path):
    with pytest.warns(RuntimeWarning, match="SECURITY WARNING"):
        sandbox = create_kernel_sandbox(
            tmp_path,
            mode="auto",
            platform_name="linux",
            which=lambda name: None,
        )

    status = sandbox.status.to_dict()
    assert status["state"] == "unavailable"
    assert status["backend"] is None
    assert status["enforced"] is False
    assert status["network_policy"] == "not_enforced"
    assert "bwrap" in status["detail"]
    assert status["warning"].startswith("OPENAI4S SECURITY WARNING")


def test_enforce_fails_closed_when_backend_is_missing(tmp_path):
    with pytest.raises(SandboxUnavailableError, match="bwrap"):
        create_kernel_sandbox(
            tmp_path,
            mode="enforce",
            platform_name="linux",
            which=lambda name: None,
        )


def test_successful_self_test_enables_seatbelt_and_private_temp(tmp_path):
    calls: list = []
    sandbox = create_kernel_sandbox(
        tmp_path,
        mode="auto",
        platform_name="darwin",
        which=lambda name: "/usr/bin/sandbox-exec",
        runner=_passing_runner(calls),
    )
    private_temp = Path(sandbox.status.temp_dir or "")
    try:
        assert sandbox.status.state == "enabled"
        assert sandbox.status.backend == "seatbelt"
        assert sandbox.status.self_test_passed is True
        assert sandbox.status.network_policy == "blocked"
        assert private_temp.is_dir()
        assert calls and calls[0][0][0] == "/usr/bin/sandbox-exec"

        env = sandbox.apply_environment({"PATH": "/usr/bin"})
        assert env["TMPDIR"] == str(private_temp)
        assert env["TMP"] == str(private_temp)
        assert env["TEMP"] == str(private_temp)
        assert sandbox.wrap_command(["/bin/true"])[0] == "/usr/bin/sandbox-exec"
    finally:
        sandbox.close()
    assert not private_temp.exists()


def test_auto_self_test_failure_falls_back_and_enforce_fails_closed(tmp_path):
    with pytest.warns(RuntimeWarning, match="self-test failed"):
        auto = create_kernel_sandbox(
            tmp_path,
            mode="auto",
            platform_name="darwin",
            which=lambda name: "/usr/bin/sandbox-exec",
            runner=_failing_runner,
        )
    assert auto.status.state == "unavailable"
    assert auto.status.backend == "seatbelt"
    assert auto.status.self_test_passed is False

    with pytest.raises(SandboxUnavailableError, match="self-test failed"):
        create_kernel_sandbox(
            tmp_path,
            mode="enforce",
            platform_name="darwin",
            which=lambda name: "/usr/bin/sandbox-exec",
            runner=_failing_runner,
        )


def test_facility_failure_and_warning_are_cached_process_wide(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def unavailable(command, **kwargs):
        del kwargs
        calls.append(list(command))
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="user namespaces are not enabled",
        )

    monkeypatch.setattr(sandbox_module, "_failed_self_tests", {})
    monkeypatch.setattr(sandbox_module, "_warned_details", set())
    monkeypatch.setattr(sandbox_module.subprocess, "run", unavailable)
    options = {
        "mode": "auto",
        "platform_name": "linux",
        "which": lambda name: "/usr/bin/bwrap",
        "runner": sandbox_module._default_runner,
    }

    with pytest.warns(RuntimeWarning, match="user namespaces are not enabled"):
        first = create_kernel_sandbox(tmp_path, **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = create_kernel_sandbox(tmp_path, **options)

    assert len(calls) == 1
    assert caught == []
    assert first.status.state == second.status.state == "unavailable"
    assert first.status.detail == second.status.detail


@pytest.mark.parametrize("value", ["maybe", "required", "TRUE-ish"])
def test_invalid_mode_is_never_silently_downgraded(tmp_path, value):
    with pytest.raises(SandboxConfigurationError, match="must be one of"):
        create_kernel_sandbox(tmp_path, mode=value)


class _RecordingSandbox:
    def __init__(self, temp_dir: Path):
        self.commands: list[list[str]] = []
        self.closed = False
        self.temp_dir = temp_dir
        self.status = SandboxStatus(
            mode="auto",
            state="enabled",
            backend="test",
            enforced=True,
            self_test_passed=True,
            network_policy="blocked",
            workspace=str(temp_dir.parent),
            temp_dir=str(temp_dir),
            detail="injected test boundary",
        )

    def wrap_command(self, command):
        self.commands.append(list(command))
        return list(command)

    def apply_environment(self, environment):
        result = dict(environment)
        result["OPENAI4S_SANDBOX_MANAGER_TEST"] = "present"
        result["TMPDIR"] = str(self.temp_dir)
        return result

    def close(self):
        self.closed = True


def test_manager_wraps_spawn_without_changing_frame_or_rpc_loop(tmp_path):
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir()
    sandbox = _RecordingSandbox(private_temp)

    with Kernel(
        dispatcher=lambda method, args: f"{method}:{args[0]}",
        cwd=str(tmp_path),
        sandbox=sandbox,
    ) as kernel:
        result = kernel.execute(
            "import os\n"
            "print(os.environ['OPENAI4S_SANDBOX_MANAGER_TEST'])\n"
            "print(host._call('echo', ['round-trip']))\n"
            "print(os.environ['TMPDIR'])"
        )
        status = kernel.sandbox_status

    assert result["error"] is None
    assert result["stdout"].splitlines() == [
        "present",
        "echo:round-trip",
        str(private_temp),
    ]
    assert len(sandbox.commands) == 1
    assert sandbox.commands[0][-1].endswith("openai4s/kernel/worker.py")
    assert status["enforced"] is True
    assert status["network_policy"] == "blocked"
    assert sandbox.closed is True


def test_raw_network_environment_flag_is_strict(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI4S_KERNEL_ALLOW_RAW_NETWORK", "sometimes")
    with pytest.raises(
        SandboxConfigurationError, match="OPENAI4S_KERNEL_ALLOW_RAW_NETWORK"
    ):
        create_kernel_sandbox(tmp_path, mode="auto")


def test_raw_network_environment_flag_is_reflected_in_status(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI4S_KERNEL_ALLOW_RAW_NETWORK", "1")
    sandbox = create_kernel_sandbox(
        tmp_path,
        mode="auto",
        platform_name="linux",
        which=lambda name: "/usr/bin/bwrap",
        runner=_passing_runner(),
    )
    try:
        assert sandbox.status.network_policy == "raw_allowed"
        assert "--unshare-net" not in sandbox.wrap_command(["/bin/true"])
    finally:
        sandbox.close()
