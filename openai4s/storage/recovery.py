"""Append-only recovery journal persistence.

Recovery is not an execution-log rewrite.  Every attempt and repair appends a
new ordered event so a failed/partial restore remains inspectable after daemon
restart.  This focused repository shares the caller's SQLite connection and
lock and has no dependency on the Store facade.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Callable

RECOVERY_JOURNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS recovery_journal (
    entry_id              TEXT PRIMARY KEY,
    recovery_id           TEXT NOT NULL,
    root_frame_id         TEXT NOT NULL,
    branch_id             TEXT NOT NULL,
    sequence              INTEGER NOT NULL CHECK (sequence >= 0),
    source_generation_id  TEXT,
    candidate_generation_id TEXT,
    phase                 TEXT NOT NULL,
    status                TEXT NOT NULL,
    detail                TEXT NOT NULL,
    created_at            INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_recovery_journal_sequence
    ON recovery_journal(recovery_id, sequence);
CREATE INDEX IF NOT EXISTS ix_recovery_journal_session
    ON recovery_journal(root_frame_id, branch_id, created_at);
"""


class RecoveryJournalRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        lock: Any,
        *,
        clock_ms: Callable[[], int],
    ) -> None:
        self._connection = connection
        self._lock = lock
        self._clock_ms = clock_ms
        with self._lock:
            self._connection.executescript(RECOVERY_JOURNAL_SCHEMA)
            self._connection.commit()

    def append(
        self,
        *,
        recovery_id: str,
        root_frame_id: str,
        branch_id: str | None,
        phase: str,
        status: str,
        detail: Any = None,
        source_generation_id: str | None = None,
        candidate_generation_id: str | None = None,
        sequence: int | None = None,
        entry_id: str | None = None,
        created_at: int | None = None,
    ) -> dict[str, Any]:
        values = {
            "recovery_id": self._text("recovery_id", recovery_id),
            "root_frame_id": self._text("root_frame_id", root_frame_id),
            "branch_id": self._text("branch_id", branch_id or root_frame_id),
            "phase": self._text("phase", phase),
            "status": self._text("status", status),
            "entry_id": self._text(
                "entry_id", entry_id or f"rj-{uuid.uuid4().hex[:16]}"
            ),
        }
        now = self._clock_ms() if created_at is None else int(created_at)
        encoded = json.dumps(
            detail if detail is not None else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            if sequence is None:
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(sequence),-1)+1 AS n "
                    "FROM recovery_journal WHERE recovery_id=?",
                    (values["recovery_id"],),
                ).fetchone()
                sequence = int(row["n"])
            if isinstance(sequence, bool) or int(sequence) < 0:
                raise ValueError("recovery sequence must be non-negative")
            self._connection.execute(
                "INSERT INTO recovery_journal("
                "entry_id,recovery_id,root_frame_id,branch_id,sequence,"
                "source_generation_id,candidate_generation_id,phase,status,"
                "detail,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    values["entry_id"],
                    values["recovery_id"],
                    values["root_frame_id"],
                    values["branch_id"],
                    int(sequence),
                    source_generation_id,
                    candidate_generation_id,
                    values["phase"],
                    values["status"],
                    encoded,
                    now,
                ),
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM recovery_journal WHERE entry_id=?",
                (values["entry_id"],),
            ).fetchone()
        return self._decode(row)

    def list(
        self,
        *,
        recovery_id: str | None = None,
        root_frame_id: str | None = None,
        branch_id: str | None = None,
        limit: int = 1000,
        newest: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("recovery_id", recovery_id),
            ("root_frame_id", root_frame_id),
            ("branch_id", branch_id),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        sql = "SELECT * FROM recovery_journal"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += (
            " ORDER BY created_at DESC,sequence DESC,entry_id DESC LIMIT ?"
            if newest
            else " ORDER BY created_at,sequence,entry_id LIMIT ?"
        )
        params.append(max(1, min(int(limit), 10_000)))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        decoded = [self._decode(row) for row in rows]
        return list(reversed(decoded)) if newest else decoded

    @staticmethod
    def _text(name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        try:
            result["detail"] = json.loads(result["detail"])
        except (TypeError, ValueError):
            pass
        return result


__all__ = ["RECOVERY_JOURNAL_SCHEMA", "RecoveryJournalRepository"]
