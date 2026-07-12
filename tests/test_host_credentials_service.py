"""Direct contracts for the session-local host credential service."""

from __future__ import annotations

import pytest

from openai4s.config import Config
from openai4s.host.credentials import CredentialService
from openai4s.host_dispatch import HostDispatcher


def test_credential_service_round_trip_overwrite_and_sorted_names():
    service = CredentialService()

    assert service.set({"name": "Z_TOKEN", "value": "z-secret"}) == {
        "ok": True,
        "name": "Z_TOKEN",
    }
    assert service.set({"name": "A_TOKEN"}) == {
        "ok": True,
        "name": "A_TOKEN",
    }
    service.set({"name": "Z_TOKEN", "value": "new-secret"})

    assert service.get("A_TOKEN") == {"name": "A_TOKEN", "value": ""}
    assert service.get("Z_TOKEN") == {
        "name": "Z_TOKEN",
        "value": "new-secret",
    }
    names = service.list()
    assert names == ["A_TOKEN", "Z_TOKEN"]
    assert "secret" not in repr(names)


def test_credential_service_preserves_key_errors_and_session_isolation():
    first = CredentialService()
    second = CredentialService()
    first.set({"name": "ONLY_FIRST", "value": "private"})

    with pytest.raises(KeyError, match="no credential 'ONLY_FIRST'"):
        second.get("ONLY_FIRST")
    with pytest.raises(KeyError, match="name"):
        first.set({"value": "missing-name"})
    assert second.list() == []


def test_credential_leases_are_action_bound_single_use_and_expiring():
    now = [100.0]
    service = CredentialService(clock=lambda: now[0])
    service.set({"name": "TOKEN", "value": "private"})

    lease = service.issue(
        "TOKEN", purpose="upload checkpoint", binding="action-a", ttl_seconds=5
    )
    assert lease["single_use"] is True
    assert "private" not in repr(lease)
    assert service.redeem(lease["token"], binding="action-a")["value"] == "private"
    with pytest.raises(KeyError, match="consumed"):
        service.redeem(lease["token"], binding="action-a")

    wrong = service.issue("TOKEN", binding="action-a")
    with pytest.raises(PermissionError, match="another action"):
        service.redeem(wrong["token"], binding="action-b")
    with pytest.raises(KeyError, match="consumed"):
        service.redeem(wrong["token"], binding="action-a")

    expired = service.issue("TOKEN", binding="action-a", ttl_seconds=1)
    now[0] += 2
    with pytest.raises(KeyError, match="expired"):
        service.redeem(expired["token"], binding="action-a")


def test_rotating_credential_revokes_outstanding_leases():
    service = CredentialService()
    service.set({"name": "TOKEN", "value": "old"})
    lease = service.issue("TOKEN")
    service.set({"name": "TOKEN", "value": "new"})
    with pytest.raises(KeyError, match="consumed"):
        service.redeem(lease["token"])
    assert service.get("TOKEN")["value"] == "new"


def test_host_dispatcher_credentials_wrappers_share_one_service(tmp_path):
    dispatcher = HostDispatcher(Config(data_dir=tmp_path))

    assert dispatcher._m_credentials_set(
        {"name": "HF_TOKEN", "value": "test-secret"}
    ) == {"ok": True, "name": "HF_TOKEN"}
    assert dispatcher._m_credentials_get("HF_TOKEN") == {
        "name": "HF_TOKEN",
        "value": "test-secret",
    }
    assert dispatcher._m_credentials_list() == ["HF_TOKEN"]
    assert dispatcher._credential_service.list() == ["HF_TOKEN"]

    lease = dispatcher._m_credentials_issue(
        {"name": "HF_TOKEN", "purpose": "one request", "ttl_seconds": 30}
    )
    assert dispatcher._m_credentials_redeem(lease["token"])["value"] == "test-secret"
    with pytest.raises(KeyError, match="consumed"):
        dispatcher._m_credentials_redeem(lease["token"])


def test_host_dispatchers_do_not_share_credentials(tmp_path):
    config = Config(data_dir=tmp_path)
    first = HostDispatcher(config)
    second = HostDispatcher(config)

    first._m_credentials_set({"name": "SESSION_ONLY", "value": "private"})
    with pytest.raises(KeyError, match="no credential 'SESSION_ONLY'"):
        second._m_credentials_get("SESSION_ONLY")


def test_dispatcher_permission_denial_precedes_credential_mutation(tmp_path):
    frame_id = "credential-frame"
    dispatcher = HostDispatcher(Config(data_dir=tmp_path), frame_id=frame_id)
    dispatcher.store.set_permission_rule(
        scope="conversation",
        scope_id=frame_id,
        tool="credentials_set",
        pattern="*",
        decision="deny",
    )

    result = dispatcher(
        "credentials_set",
        [{"name": "BLOCKED_TOKEN", "value": "must-not-be-stored"}],
    )

    assert set(result) == {"error"}
    assert "Permission denied" in result["error"]
    assert dispatcher._credential_service.list() == []


def test_replay_excludes_all_credential_methods_and_values(tmp_path):
    from openai4s.replay import TapeRecorder

    dispatcher = HostDispatcher(Config(data_dir=tmp_path))
    # The production default for injecting a credential is deliberately
    # ``ask``.  This headless replay contract is testing redaction rather than
    # unattended approval, so authorize the mutation explicitly.
    dispatcher.store.set_permission_rule(
        scope="global",
        scope_id="",
        tool="credentials_set",
        pattern="*",
        decision="allow",
    )
    recorder = TapeRecorder(tmp_path / "credentials-tape.json")
    dispatcher.recorder = recorder
    secret = "synthetic-secret-never-record"

    dispatcher("credentials_set", [{"name": "TOKEN", "value": secret}])
    assert dispatcher("credentials_get", ["TOKEN"])["value"] == secret
    lease = dispatcher(
        "credentials_issue",
        [{"name": "TOKEN", "purpose": "single use"}],
    )
    assert dispatcher("credentials_redeem", [lease["token"]])["value"] == secret
    assert dispatcher("credentials_list", []) == ["TOKEN"]

    assert not {
        "credentials_set",
        "credentials_get",
        "credentials_issue",
        "credentials_redeem",
        "credentials_list",
    } & {record["method"] for record in recorder.records}
    assert secret not in repr(recorder.records)
