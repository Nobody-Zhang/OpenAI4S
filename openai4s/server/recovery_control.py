"""Recovery journal projection and verified pipeline composition.

The kernel recovery algorithm is intentionally callback-driven.  This service
binds its journal to durable Store ports, exposes a small safe status view, and
describes which recovery actions are currently possible.  It never claims a
checkpoint is restorable unless both a workspace tree and a complete bootstrap
manifest are present.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Callable, Protocol

from openai4s.kernel.recovery import KernelRecoveryOrchestrator

_STATUSES = frozenset(
    {"started", "completed", "skipped", "partial", "failed", "cancelled"}
)


class RecoveryStore(Protocol):
    def append_recovery_event(self, **fields: Any) -> dict: ...

    def list_recovery_events(self, **filters: Any) -> list[dict]: ...

    def get_session_branch(self, branch_id: str) -> dict | None: ...

    def get_session_checkpoint(self, checkpoint_id: str) -> dict | None: ...

    def latest_kernel_generation(
        self, root_frame_id: str, language: str, *, branch_id: str | None = None
    ) -> dict | None: ...


class RecoveryControlService:
    """Durable recovery status/actions with a journal-bound pipeline factory."""

    def __init__(
        self,
        store: RecoveryStore,
        *,
        workspace_tree_exists: Callable[[str], bool] | None = None,
        payload_chars: int = 20_000,
    ) -> None:
        if payload_chars < 256:
            raise ValueError("payload_chars must be at least 256")
        self.store = store
        self._workspace_tree_exists = workspace_tree_exists or (lambda _tree_id: False)
        self.payload_chars = payload_chars

    def record(self, event: Mapping[str, Any]) -> dict:
        """Append one orchestrator journal event without accepting opaque fields."""

        required = ("recovery_id", "root_frame_id", "phase", "status")
        missing = [name for name in required if not str(event.get(name) or "").strip()]
        if missing:
            raise ValueError("recovery event missing: " + ", ".join(missing))
        phase = str(event["phase"]).strip().lower()
        status = str(event["status"]).strip().lower()
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase) is None:
            raise ValueError(f"invalid recovery phase: {phase!r}")
        if status not in _STATUSES:
            raise ValueError(f"unknown recovery status: {status!r}")
        return self.store.append_recovery_event(
            recovery_id=str(event["recovery_id"]),
            root_frame_id=str(event["root_frame_id"]),
            branch_id=(
                str(event["branch_id"]) if event.get("branch_id") else None
            ),
            source_generation_id=(
                str(event["source_generation_id"])
                if event.get("source_generation_id")
                else None
            ),
            candidate_generation_id=(
                str(event["candidate_generation_id"])
                if event.get("candidate_generation_id")
                else None
            ),
            phase=phase,
            status=status,
            # Recovery errors can contain echoed environment/HTTP details. Do
            # not persist credential-shaped fields merely to redact them later.
            detail=_redact(event.get("detail") or {}),
        )

    def pipeline(self, **ports: Any) -> KernelRecoveryOrchestrator:
        """Build a recovery pipeline whose every phase is durably journaled."""

        if "journal" in ports:
            raise ValueError("recovery journal is owned by RecoveryControlService")
        return KernelRecoveryOrchestrator(journal=self.record, **ports)

    def status(
        self,
        root_frame_id: str,
        *,
        branch_id: str | None = None,
        recovery_id: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        root_frame_id = _required("root_frame_id", root_frame_id)
        branch_id = branch_id or root_frame_id
        events = self.store.list_recovery_events(
            recovery_id=recovery_id,
            root_frame_id=root_frame_id,
            branch_id=branch_id,
            limit=limit,
            newest=True,
        )
        grouped: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            grouped[str(event.get("recovery_id") or "")].append(event)
        attempts = [self._attempt(rows) for rows in grouped.values() if rows]
        attempts.sort(
            key=lambda item: (
                item.get("updated_at") or 0,
                item.get("recovery_id") or "",
            ),
            reverse=True,
        )
        generations = {
            language: self.store.latest_kernel_generation(
                root_frame_id,
                language,
                branch_id=branch_id,
            )
            for language in ("python", "r")
        }
        current = attempts[0] if attempts else None
        state = (
            current["state"]
            if current is not None
            else _generation_state(generations.values())
        )
        return {
            "root_frame_id": root_frame_id,
            "branch_id": branch_id,
            "state": state,
            "current": current,
            "attempts": attempts,
            "generations": {
                key: _public_generation(value) for key, value in generations.items()
            },
        }

    def actions(
        self,
        root_frame_id: str,
        *,
        branch_id: str | None = None,
    ) -> dict[str, Any]:
        root_frame_id = _required("root_frame_id", root_frame_id)
        branch_id = branch_id or root_frame_id
        status = self.status(root_frame_id, branch_id=branch_id)
        branch = self.store.get_session_branch(branch_id)
        if branch is not None and branch.get("root_frame_id") != root_frame_id:
            raise PermissionError("branch belongs to another session")
        checkpoint = (
            self.store.get_session_checkpoint(branch.get("head_checkpoint_id"))
            if branch and branch.get("head_checkpoint_id")
            else None
        )
        restorable, unavailable = _restorable(checkpoint)
        if restorable:
            try:
                tree_exists = self._workspace_tree_exists(
                    str(checkpoint.get("workspace_tree_id"))
                )
            except Exception:  # noqa: BLE001 - CAS verification fails closed
                tree_exists = False
            if not tree_exists:
                restorable = False
                unavailable = "checkpoint workspace tree is missing or corrupt"
        busy = status["state"] in {
            "restoring",
            "bootstrapping",
            "hydrating",
            "replaying",
            "validating",
        }
        recoverable_state = status["state"] in {
            "none",
            "ended",
            "partial",
            "failed",
        }
        latest_state = (status.get("current") or {}).get("state")
        actions = [
            _action(
                "restore",
                restorable and recoverable_state and not busy,
                (
                    unavailable
                    if not restorable
                    else "recovery already running"
                    if busy
                    else "kernel is already active"
                    if not recoverable_state
                    else None
                ),
                requires_ticket=True,
            ),
            _action(
                "retry",
                restorable and latest_state in {"partial", "failed"} and not busy,
                (
                    unavailable
                    if not restorable
                    else "latest recovery is not partial or failed"
                    if latest_state not in {"partial", "failed"}
                    else "recovery already running"
                    if busy
                    else None
                ),
                requires_ticket=True,
            ),
            _action("inspect_log", True, None),
            _action("continue_view_only", True, None),
            _action(
                "restart_fresh",
                not busy,
                "recovery already running" if busy else None,
                requires_ticket=True,
                requires_confirmation=True,
            ),
        ]
        return {
            "root_frame_id": root_frame_id,
            "branch_id": branch_id,
            "checkpoint_id": (
                checkpoint.get("checkpoint_id") if checkpoint else None
            ),
            "state": status["state"],
            "actions": actions,
        }

    def _attempt(self, rows: list[dict]) -> dict[str, Any]:
        rows.sort(
            key=lambda item: (
                item.get("created_at") or 0,
                item.get("sequence") or 0,
                item.get("entry_id") or "",
            )
        )
        latest = rows[-1]
        return {
            "recovery_id": latest.get("recovery_id"),
            "state": _journal_state(rows),
            "phase": latest.get("phase"),
            "phase_status": latest.get("status"),
            "source_generation_id": latest.get("source_generation_id"),
            "candidate_generation_id": latest.get("candidate_generation_id"),
            "started_at": rows[0].get("created_at"),
            "updated_at": latest.get("created_at"),
            "events": [
                {
                    "entry_id": row.get("entry_id"),
                    "sequence": row.get("sequence"),
                    "phase": row.get("phase"),
                    "status": row.get("status"),
                    "detail": _bounded_public(row.get("detail"), self.payload_chars),
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
        }


def _journal_state(rows: list[dict]) -> str:
    latest = rows[-1]
    phase = str(latest.get("phase") or "")
    status = str(latest.get("status") or "")
    if phase == "publish" and status == "completed":
        return "active"
    if status == "partial":
        return "partial"
    if status in {"failed", "cancelled"}:
        return "failed"
    phase_states = {
        "build": "bootstrapping",
        "bootstrap": "bootstrapping",
        "hydrate_workspace": "hydrating",
        "hydrate_artifact": "hydrating",
        "replay": "replaying",
        "validate": "validating",
    }
    return phase_states.get(phase, "restoring")


def _generation_state(generations) -> str:
    states = {
        str(item.get("state") or "")
        for item in generations
        if isinstance(item, Mapping)
    }
    if states & {"active", "busy"}:
        return "active"
    if states & {"partial"}:
        return "partial"
    if states & {"failed", "crashed"}:
        return "failed"
    return "ended" if states else "none"


def _public_generation(value: Mapping[str, Any] | None) -> dict | None:
    if value is None:
        return None
    return {
        key: value.get(key)
        for key in (
            "generation_id",
            "language",
            "ordinal",
            "state",
            "bootstrap_manifest_id",
            "environment_manifest_id",
            "last_activity_at",
            "ended_at",
            "ended_reason",
            "recovered_from_generation_id",
        )
    }


def _restorable(checkpoint: Mapping[str, Any] | None) -> tuple[bool, str | None]:
    if checkpoint is None:
        return False, "no checkpoint exists"
    if not checkpoint.get("workspace_tree_id"):
        return False, "checkpoint has no workspace tree"
    refs = checkpoint.get("generation_refs")
    if not isinstance(refs, Mapping) or not refs:
        return False, "checkpoint has no verifiable bootstrap manifest"
    for language, value in refs.items():
        manifest = (
            value.get("bootstrap_manifest") or value.get("bootstrap")
            if isinstance(value, Mapping)
            else None
        )
        if not isinstance(manifest, Mapping) or int(manifest.get("version") or 0) != 1:
            return False, f"checkpoint lacks a verifiable {language} bootstrap manifest"
    recipe = checkpoint.get("recovery_recipe")
    if not isinstance(recipe, Mapping):
        return False, "checkpoint has no recovery recipe"
    return True, None


def _action(
    action_id: str,
    enabled: bool,
    reason: str | None,
    *,
    requires_ticket: bool = False,
    requires_confirmation: bool = False,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "enabled": bool(enabled),
        "reason": reason,
        "requires_execution_ticket": requires_ticket,
        "requires_confirmation": requires_confirmation,
    }


def _bounded_public(value: Any, limit: int) -> Any:
    safe = _redact(value)
    encoded = json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    if len(encoded) <= limit:
        return safe
    return {
        "truncated": True,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "original_chars": len(encoded),
        "preview": encoded[: limit - 1] + "…",
    }


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if any(
                    marker in str(key).casefold()
                    for marker in (
                        "secret",
                        "token",
                        "password",
                        "credential",
                        "api_key",
                    )
                )
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _required(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


__all__ = ["RecoveryControlService", "RecoveryStore"]
