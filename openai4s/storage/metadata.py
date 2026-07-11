"""Small, cohesive metadata repositories on a Store-owned SQLite connection."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Callable

# Credential reads are derivable and must never be duplicated in the audit log.
DERIVABLE_HOST_CALLS = frozenset({"credentials_get", "credentials_list"})

# Credential writes remain auditable by method name, but their raw arguments do
# not cross the persistence boundary.
SECRET_ARG_HOST_CALLS = frozenset({"credentials_set"})


class NotesRepository:
    """Persist project notes and expose their legacy API projection."""

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

    def add(
        self,
        *,
        project_id: str,
        content: str,
        title: str | None = None,
    ) -> dict:
        now = self._clock_ms()
        note_id = f"note_{uuid.uuid4().hex[:12]}"
        self._execute(
            "INSERT INTO notes(note_id,project_id,title,body,created_at) "
            "VALUES(?,?,?,?,?)",
            (note_id, project_id, title, content, now),
        )
        return {
            "note_id": note_id,
            "project_id": project_id,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }

    def list(self, project_id: str) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT note_id,project_id,title,body,created_at FROM notes "
                "WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [
            {
                "note_id": row["note_id"],
                "project_id": row["project_id"],
                "content": row["body"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, note_id: str) -> None:
        self._execute("DELETE FROM notes WHERE note_id=?", (note_id,))

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._connection.execute(sql, params)
            self._connection.commit()


class FolderRepository:
    """Persist project folders and frame-to-folder assignments."""

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

    def create(self, *, project_id: str, name: str) -> dict:
        now = self._clock_ms()
        folder_id = f"fold_{uuid.uuid4().hex[:10]}"
        self._execute(
            "INSERT INTO folders(folder_id,project_id,name,created_at) "
            "VALUES(?,?,?,?)",
            (folder_id, project_id, name, now),
        )
        return {
            "folder_id": folder_id,
            "project_id": project_id,
            "name": name,
            "created_at": now,
        }

    def list(self, project_id: str) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT folder_id,project_id,name,created_at FROM folders "
                "WHERE project_id=? ORDER BY name",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename(self, folder_id: str, name: str) -> None:
        self._execute(
            "UPDATE folders SET name=? WHERE folder_id=?",
            (name, folder_id),
        )

    def delete(self, folder_id: str) -> None:
        # Keep the historical two-transaction boundary: frames are un-filed and
        # committed before the folder row is removed in a second transaction.
        self._execute(
            "UPDATE frames SET folder_id=NULL WHERE folder_id=?",
            (folder_id,),
        )
        self._execute(
            "DELETE FROM folders WHERE folder_id=?",
            (folder_id,),
        )

    def set_frame_folder(self, frame_id: str, folder_id: str | None) -> None:
        self._execute(
            "UPDATE frames SET folder_id=? WHERE frame_id=?",
            (folder_id, frame_id),
        )

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._connection.execute(sql, params)
            self._connection.commit()


class EndpointRepository:
    """Persist managed endpoint metadata with legacy dynamic-field upserts."""

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

    def upsert(self, name: str, **fields: Any) -> None:
        now = self._clock_ms()
        with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM managed_endpoints WHERE name=?",
                (name,),
            ).fetchone()
            if exists:
                fields["updated_at"] = now
                columns = ", ".join(f"{key}=?" for key in fields)
                self._connection.execute(
                    f"UPDATE managed_endpoints SET {columns} WHERE name=?",
                    (*fields.values(), name),
                )
            else:
                fields.setdefault("created_at", now)
                fields["updated_at"] = now
                fields["name"] = name
                columns = ", ".join(fields)
                placeholders = ", ".join("?" for _ in fields)
                self._connection.execute(
                    f"INSERT INTO managed_endpoints({columns}) "
                    f"VALUES({placeholders})",
                    tuple(fields.values()),
                )
            self._connection.commit()

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM managed_endpoints ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]


class CompactionRepository:
    """Archive compacted conversation slices for later inspection."""

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

    def archive(
        self,
        *,
        frame_id: str | None,
        summary: str,
        compacted: list[dict],
        project_id: str = "default",
    ) -> str:
        archive_id = f"ca-{uuid.uuid4().hex[:12]}"
        self._execute(
            "INSERT INTO compaction_archives(archive_id,frame_id,project_id,"
            "summary,compacted,n_messages,created_at) VALUES(?,?,?,?,?,?,?)",
            (
                archive_id,
                frame_id,
                project_id,
                summary,
                json.dumps(compacted, ensure_ascii=False),
                len(compacted),
                self._clock_ms(),
            ),
        )
        return archive_id

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._connection.execute(sql, params)
            self._connection.commit()


class HostCallRepository:
    """Persist scrubbed host-RPC audit records."""

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

    def log(
        self,
        *,
        method: str,
        args: list,
        ok: bool,
        frame_id: str | None = None,
    ) -> None:
        if method in DERIVABLE_HOST_CALLS:
            return
        if method in SECRET_ARG_HOST_CALLS:
            preview = "<redacted secret args>"
        else:
            try:
                preview = json.dumps(args, ensure_ascii=False)[:500]
            except (TypeError, ValueError):
                preview = "<unserializable>"
        self._execute(
            "INSERT INTO host_call_log(call_id,frame_id,method,args_preview,ok,"
            "created_at) VALUES(?,?,?,?,?,?)",
            (
                f"hc-{uuid.uuid4().hex[:12]}",
                frame_id,
                method,
                preview,
                1 if ok else 0,
                self._clock_ms(),
            ),
        )

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._connection.execute(sql, params)
            self._connection.commit()


__all__ = [
    "CompactionRepository",
    "DERIVABLE_HOST_CALLS",
    "EndpointRepository",
    "FolderRepository",
    "HostCallRepository",
    "NotesRepository",
    "SECRET_ARG_HOST_CALLS",
]
