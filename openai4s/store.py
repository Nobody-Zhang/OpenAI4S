"""SQLite data model — the shared store the whole system records into.

openai4s exposes these tables read-only through `host.query`; the host writes them
as turns/cells/artifacts/compactions happen. Schema and write paths:

  frames            turn tree (self-referential), per-turn model/effort/token/cost
  execution_log     per-cell record (code + usage wall/cpu/rss + error)
  artifacts         logical artifact (filename, content_type)
  artifact_versions versioned bytes (version_id, checksum) -> artifacts
  compaction_archives  compacted history slices
  agents            agent profile definitions
  custom_skills     user-authored SKILL.md bodies
  memories          memory blocks (scope/block-listed in host.query)
  managed_endpoints local model endpoints
  notes             project notes
  lineage_edges     object-level data lineage: input_version -> output_version
  host_call_log     RPC audit (DERIVABLE_HOST_CALLS are NOT logged; the args of
                    SECRET_ARG_HOST_CALLS are redacted before write)

Secret-bearing tables (`settings` holds the LLM API key + model profiles,
`connectors` holds MCP server env/command) plus the internal audit/memory tables
are on QUERY_DENYLIST, so `host.query` refuses to read them and `host.query`'s
schema view hides them.

All timestamps are epoch-ms. Booleans are 0/1. One DB per data_dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from openai4s.storage.agents import AgentProfileRepository
from openai4s.storage.annotations import AnnotationRepository
from openai4s.storage.connectors import ConnectorRepository
from openai4s.storage.frames import FrameRepository
from openai4s.storage.memories import MemoryRepository
from openai4s.storage.metadata import (
    DERIVABLE_HOST_CALLS,
    SECRET_ARG_HOST_CALLS,
    CompactionRepository,
    EndpointRepository,
    FolderRepository,
    HostCallRepository,
    NotesRepository,
)
from openai4s.storage.permissions import (
    DEFAULT_PERMISSION_RULES as _DEFAULT_PERMISSION_RULES,
)
from openai4s.storage.permissions import PermissionRuleRepository
from openai4s.storage.permissions import perm_match as _perm_match
from openai4s.storage.plans import PlanRepository
from openai4s.storage.settings import SettingsRepository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id      TEXT PRIMARY KEY,
    parent_id     TEXT,
    project_id    TEXT NOT NULL DEFAULT 'default',
    root_frame_id TEXT,
    kind          TEXT,               -- 'turn' | 'delegate' | 'compaction_fork'
    name          TEXT,
    task_summary  TEXT,               -- auto one-line summary shown in the UI
    model         TEXT,
    effort        TEXT,
    status        TEXT,               -- 'processing'|'done'|'failed'|'awaiting_user_response'
    runtime_env   TEXT,
    depth         INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_frames_parent  ON frames(parent_id);
CREATE INDEX IF NOT EXISTS ix_frames_project ON frames(project_id);

CREATE TABLE IF NOT EXISTS projects (
    project_id    TEXT PRIMARY KEY,
    name          TEXT,
    description   TEXT,
    context       TEXT,               -- agent context prepended to prompts
    is_example    INTEGER NOT NULL DEFAULT 0,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id    TEXT PRIMARY KEY,
    root_frame_id TEXT NOT NULL,
    frame_id      TEXT,
    seq           INTEGER NOT NULL,
    role          TEXT NOT NULL,      -- 'user' | 'assistant'
    content       TEXT,               -- plain text (may be markdown)
    metadata      TEXT,               -- JSON blob (optional)
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_msg_root ON messages(root_frame_id);

CREATE TABLE IF NOT EXISTS execution_log (
    producing_cell_id TEXT PRIMARY KEY,
    frame_id      TEXT,
    root_frame_id TEXT,
    project_id    TEXT NOT NULL DEFAULT 'default',
    cell_seq      INTEGER,
    cell_index    INTEGER,
    kernel_id     TEXT,
    language      TEXT,
    status        TEXT,
    origin        TEXT,
    code          TEXT NOT NULL,
    stdout        TEXT,
    stderr        TEXT,
    error         TEXT,
    figures       TEXT,               -- JSON list of artifact filenames
    files_read    TEXT,               -- JSON list of relative paths
    files_written TEXT,               -- JSON list of relative paths
    interrupted   INTEGER NOT NULL DEFAULT 0,
    wall_s        REAL,
    cpu_s         REAL,
    peak_rss_kb   INTEGER,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exec_frame ON execution_log(frame_id);
CREATE INDEX IF NOT EXISTS ix_exec_root  ON execution_log(root_frame_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL DEFAULT 'default',
    root_frame_id TEXT,
    filename      TEXT NOT NULL,
    content_type  TEXT,
    is_user_upload INTEGER NOT NULL DEFAULT 0,
    priority      INTEGER NOT NULL DEFAULT 0,
    latest_version_id TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_versions (
    version_id    TEXT PRIMARY KEY,
    artifact_id   TEXT NOT NULL,
    filename      TEXT,
    content_type  TEXT,
    size_bytes    INTEGER,
    checksum      TEXT,
    path          TEXT NOT NULL,
    snapshot_path TEXT,
    producing_cell_id TEXT,
    frame_id      TEXT,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ver_artifact ON artifact_versions(artifact_id);

-- De-duplicated environment snapshots (one row per distinct kernel env). An
-- artifact_version references one via env_snapshot_id so a figure records the
-- package set that PRODUCED it (see gateway._environment_snapshot).
CREATE TABLE IF NOT EXISTS env_snapshots (
    snapshot_id    TEXT PRIMARY KEY,
    created_at     INTEGER NOT NULL,
    kind           TEXT,
    python_version TEXT,
    implementation TEXT,
    platform       TEXT,
    package_count  INTEGER,
    packages_json  TEXT,
    remote_json    TEXT               -- JSON list of remote-GPU job provenance
);

CREATE TABLE IF NOT EXISTS compaction_archives (
    archive_id    TEXT PRIMARY KEY,
    frame_id      TEXT,
    project_id    TEXT NOT NULL DEFAULT 'default',
    summary       TEXT,
    compacted     TEXT,               -- JSON of the raw slice
    n_messages    INTEGER,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    name          TEXT PRIMARY KEY,   -- UPPER_SNAKE (2-32)
    description   TEXT,
    skill_names   TEXT,               -- JSON list or NULL (=unrestricted)
    connectors    TEXT,               -- JSON list
    unrestricted  INTEGER NOT NULL DEFAULT 1,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_skills (
    name          TEXT PRIMARY KEY,
    origin        TEXT,
    skill_md      TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id     TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL DEFAULT 'default',
    block         TEXT,               -- memory block name
    content       TEXT,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS managed_endpoints (
    name          TEXT PRIMARY KEY,
    url           TEXT,
    skill         TEXT,
    port          INTEGER,
    status        TEXT,               -- 'registered'|'starting'|'live'|'stopped'
    credential    TEXT,
    start_script  TEXT,
    stop_script   TEXT,
    live_route    TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    note_id       TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL DEFAULT 'default',
    title         TEXT,
    body          TEXT,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS lineage_edges (
    edge_id           TEXT PRIMARY KEY,
    input_version_id  TEXT NOT NULL,
    output_version_id TEXT NOT NULL,
    producing_cell_id TEXT,
    frame_id          TEXT,
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_edge_out ON lineage_edges(output_version_id);
CREATE INDEX IF NOT EXISTS ix_edge_in  ON lineage_edges(input_version_id);

CREATE TABLE IF NOT EXISTS host_call_log (
    call_id       TEXT PRIMARY KEY,
    frame_id      TEXT,
    method        TEXT NOT NULL,
    args_preview  TEXT,
    ok            INTEGER NOT NULL DEFAULT 1,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key           TEXT PRIMARY KEY,
    value         TEXT,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS folders (
    folder_id     TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL DEFAULT 'default',
    name          TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS connectors (
    connector_id  TEXT PRIMARY KEY,   -- slug
    name          TEXT NOT NULL,
    description   TEXT,
    command       TEXT NOT NULL,      -- JSON list argv OR a shell string
    args          TEXT,               -- JSON list
    env           TEXT,               -- JSON dict
    enabled       INTEGER NOT NULL DEFAULT 1,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS frame_steps (
    step_id       TEXT PRIMARY KEY,
    frame_id      TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    kind          TEXT NOT NULL,      -- search|plan|env|skill|bash|edit|write|read|files|artifact|delegate|mcp|fetch|code
    title         TEXT,
    summary       TEXT,               -- one-line result summary (shown as meta)
    input         TEXT,               -- JSON
    output        TEXT,               -- JSON
    status        TEXT,               -- running|done|error
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frame_steps_frame ON frame_steps(frame_id, seq);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  TEXT PRIMARY KEY,
    root_frame_id  TEXT NOT NULL,
    artifact_id    TEXT NOT NULL,
    artifact_name  TEXT,
    rel_x          REAL NOT NULL,      -- 0..1 fraction of image width
    rel_y          REAL NOT NULL,      -- 0..1 fraction of image height
    number         INTEGER NOT NULL,   -- pin ordinal within (frame,artifact)
    body           TEXT NOT NULL,      -- the comment
    status         TEXT NOT NULL DEFAULT 'open',   -- open|sent|resolved
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_annot_frame    ON annotations(root_frame_id);
CREATE INDEX IF NOT EXISTS ix_annot_artifact ON annotations(artifact_id);

CREATE TABLE IF NOT EXISTS plans (
    plan_id       TEXT PRIMARY KEY,
    frame_id      TEXT NOT NULL,
    project_id    TEXT NOT NULL DEFAULT 'default',
    title         TEXT,
    rationale     TEXT,
    confidence    TEXT,               -- 'high'|'medium'|'low' (or a 0..1 string)
    steps         TEXT NOT NULL,      -- JSON [{id,title,detail,deliverables:[...]}]
    status        TEXT NOT NULL DEFAULT 'draft',   -- draft|executing|completed|failed|discarded
    step_status   TEXT,               -- JSON {step_id: {status, note, updated_at}}
    artifact_id   TEXT,               -- the plan_*.json artifact (so revises re-version it)
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_frame ON plans(frame_id, created_at);

-- opencode-style tool-call permission rules. Each rule maps a (tool, pattern)
-- to allow|ask|deny at one of three scopes: 'global' (scope_id=''),
-- 'project' (scope_id=project_id) or 'conversation' (scope_id=root_frame_id).
CREATE TABLE IF NOT EXISTS permission_rules (
    rule_id       TEXT PRIMARY KEY,
    scope         TEXT NOT NULL,               -- global | project | conversation
    scope_id      TEXT NOT NULL DEFAULT '',    -- '' for global; project_id; root_frame_id
    tool          TEXT NOT NULL,               -- host method name, or '*'
    pattern       TEXT NOT NULL DEFAULT '*',   -- glob matched against the tool target
    decision      TEXT NOT NULL,               -- allow | ask | deny
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_perm ON permission_rules(scope, scope_id, tool, pattern);
CREATE INDEX IF NOT EXISTS ix_perm_scope ON permission_rules(scope, scope_id);
"""

# Tables host.query must refuse to read. These hold secrets or
# internal audit/memory state that is not part of the agent-visible data model:
#   settings          -> LLM API key + model profiles (which embed API keys)
#   connectors        -> MCP server env vars / launch command (may embed tokens)
#   memories          -> memory blocks (surfaced through host.remember, not SQL)
#   host_call_log     -> RPC audit trail
#   permission_rules  -> permission broker state
QUERY_DENYLIST = frozenset(
    {"settings", "connectors", "memories", "host_call_log", "permission_rules"}
)

# Single-quoted string literals and SQL comments are stripped before the denylist
# substring test so a denied table name that appears only inside a *literal*
# (e.g. SELECT 'see settings' AS note) is not falsely rejected — a real table
# reference can never live inside a string literal. Double-quoted / bracketed /
# backtick spans are left intact because SQL uses them to quote identifiers
# (e.g. FROM "settings"), which must still trip the denylist.
_SQL_LITERAL_RE = re.compile(
    r"'(?:[^']|'')*'"  # single-quoted string (with '' escape)
    r"|--[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    re.DOTALL,
)


def _strip_sql_literals(sql: str) -> str:
    """Blank out single-quoted string literals and comments for denylist checks."""
    return _SQL_LITERAL_RE.sub(" ", sql or "")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _file_identity(path: str) -> str | None:
    """Best-effort physical identity for legacy/aliased artifact paths."""
    try:
        raw = os.fsdecode(os.fspath(path))
        return os.path.normcase(os.path.realpath(raw))
    except (TypeError, ValueError, OSError):
        return None


def _same_file_path(left: str, right: str) -> bool:
    """Return whether two stored paths identify the same physical file."""
    if left == right:
        return True
    left_identity = _file_identity(left)
    right_identity = _file_identity(right)
    return (
        left_identity is not None
        and right_identity is not None
        and left_identity == right_identity
    )


class Store:
    """Thread-safe SQLite wrapper. One per data_dir; created lazily."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()
        self._plans = PlanRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._annotations = AnnotationRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._memories = MemoryRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._settings = SettingsRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._permissions = PermissionRuleRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
            get_setting=self.get_setting,
            set_setting=self.set_setting,
        )
        self._connectors = ConnectorRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._agents = AgentProfileRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._frames = FrameRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
            get_frame=lambda frame_id: self.get_frame(frame_id),
            resolve_frame_scope=lambda frame_id, **kwargs: self.resolve_frame_scope(
                frame_id, **kwargs
            ),
            get_project=lambda project_id: self.get_project(project_id),
        )
        self._notes = NotesRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._folders = FolderRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._endpoints = EndpointRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._compactions = CompactionRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )
        self._host_calls = HostCallRepository(
            self._conn,
            self._lock,
            clock_ms=lambda: _now_ms(),
        )

    # --- migration (add columns missing from a pre-existing DB) -----------
    _MIGRATIONS = {
        "frames": [
            ("task_summary", "TEXT"),
            ("folder_id", "TEXT"),
            ("runtime_env", "TEXT"),
        ],
        "agents": [("system_prompt", "TEXT"), ("kind", "TEXT")],
        "artifact_versions": [("env_snapshot_id", "TEXT"), ("snapshot_path", "TEXT")],
        "env_snapshots": [("remote_json", "TEXT")],
        "execution_log": [
            ("root_frame_id", "TEXT"),
            ("cell_index", "INTEGER"),
            ("kernel_id", "TEXT"),
            ("language", "TEXT"),
            ("status", "TEXT"),
            ("figures", "TEXT"),
            ("files_read", "TEXT"),
            ("files_written", "TEXT"),
        ],
    }

    def _migrate(self) -> None:
        with self._lock:
            for table, cols in self._MIGRATIONS.items():
                have = {
                    r["name"]
                    for r in self._conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                for name, decl in cols:
                    if name not in have:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE {table} ADD COLUMN {name} {decl}"
                            )
                        except sqlite3.OperationalError:
                            pass
            # Historical child frames inherited the root id but silently kept
            # project_id='default'. Historical artifacts also used their actor
            # frame as root_frame_id. Repair both idempotently when the frame
            # tree still exists; unframed legacy uploads remain untouched.
            self._conn.execute(
                "UPDATE frames SET project_id=COALESCE((SELECT root.project_id "
                "FROM frames AS root WHERE root.frame_id=frames.root_frame_id),"
                "project_id) WHERE root_frame_id IS NOT NULL"
            )
            self._conn.execute(
                "UPDATE artifacts SET project_id=COALESCE((SELECT root.project_id "
                "FROM frames AS actor JOIN frames AS root "
                "ON root.frame_id=actor.root_frame_id "
                "WHERE actor.frame_id=artifacts.root_frame_id),project_id) "
                "WHERE root_frame_id IN (SELECT frame_id FROM frames)"
            )
            self._conn.execute(
                "UPDATE artifacts SET root_frame_id=COALESCE((SELECT "
                "actor.root_frame_id FROM frames AS actor "
                "WHERE actor.frame_id=artifacts.root_frame_id),root_frame_id) "
                "WHERE root_frame_id IN (SELECT frame_id FROM frames)"
            )
            self._conn.commit()

    # --- low-level -------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- frames ----------------------------------------------------------
    def new_frame(
        self,
        *,
        parent_id: str | None = None,
        project_id: str = "default",
        kind: str = "turn",
        name: str | None = None,
        model: str | None = None,
        depth: int = 0,
        status: str = "processing",
    ) -> str:
        return self._frames.new_frame(
            parent_id=parent_id,
            project_id=project_id,
            kind=kind,
            name=name,
            model=model,
            depth=depth,
            status=status,
        )

    def resolve_frame_scope(
        self,
        frame_id: str | None,
        *,
        fallback_project: str = "default",
    ) -> dict:
        return self._frames.resolve_frame_scope(
            frame_id,
            fallback_project=fallback_project,
        )

    def update_frame(self, frame_id: str, **fields: Any) -> None:
        self._frames.update_frame(frame_id, **fields)

    def add_frame_tokens(
        self,
        frame_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self._frames.add_frame_tokens(
            frame_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    # --- projects --------------------------------------------------------
    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        context: str = "",
        project_id: str | None = None,
        is_example: bool = False,
    ) -> dict:
        return self._frames.create_project(
            name=name,
            description=description,
            context=context,
            project_id=project_id,
            is_example=is_example,
        )

    def get_project(self, project_id: str) -> dict | None:
        return self._frames.get_project(project_id)

    def update_project(self, project_id: str, **fields: Any) -> None:
        self._frames.update_project(project_id, **fields)

    def delete_project(self, project_id: str) -> dict:
        return self._frames.delete_project(project_id)

    def list_projects(self) -> list[dict]:
        return self._frames.list_projects()

    # --- messages --------------------------------------------------------
    def add_message(
        self,
        *,
        root_frame_id: str,
        role: str,
        content: str,
        frame_id: str | None = None,
        metadata: dict | None = None,
        created_at: int | None = None,
    ) -> dict:
        return self._frames.add_message(
            root_frame_id=root_frame_id,
            role=role,
            content=content,
            frame_id=frame_id,
            metadata=metadata,
            created_at=created_at,
        )

    def list_messages(
        self, root_frame_id: str, *, start: int = 0, limit: int = 300
    ) -> list[dict]:
        return self._frames.list_messages(
            root_frame_id,
            start=start,
            limit=limit,
        )

    def message_count(self, root_frame_id: str) -> int:
        return self._frames.message_count(root_frame_id)

    def cell_count(self, root_frame_id: str) -> int:
        return self._frames.cell_count(root_frame_id)

    # --- semantic activity steps (plan / search / env / skill / edit / …) ----
    # Every visible host.* tool call becomes a persisted "step" so a reopened
    # session re-renders the same rich activity (not just the final prose).
    def add_step(
        self,
        *,
        step_id: str,
        frame_id: str,
        kind: str,
        title: str | None = None,
        input: dict | None = None,
        status: str = "running",
    ) -> dict:
        return self._frames.add_step(
            step_id=step_id,
            frame_id=frame_id,
            kind=kind,
            title=title,
            input=input,
            status=status,
        )

    def update_step(
        self,
        step_id: str,
        *,
        status: str | None = None,
        output: dict | None = None,
        title: str | None = None,
        summary: str | None = None,
    ) -> None:
        self._frames.update_step(
            step_id,
            status=status,
            output=output,
            title=title,
            summary=summary,
        )

    def list_steps(
        self, frame_id: str, *, start: int = 0, limit: int = 800
    ) -> list[dict]:
        return self._frames.list_steps(frame_id, start=start, limit=limit)

    def step_count(self, frame_id: str) -> int:
        return self._frames.step_count(frame_id)

    # --- frame browse / detail / search --------------------------
    def browse_frames(
        self,
        *,
        project_id: str | None = "default",
        status: str | None = None,
        roots_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        return self._frames.browse_frames(
            project_id=project_id,
            status=status,
            roots_only=roots_only,
            limit=limit,
        )

    def frame_detail(
        self, frame_id: str, *, page: int = 0, page_size: int = 50
    ) -> dict | None:
        return self._frames.frame_detail(
            frame_id,
            page=page,
            page_size=page_size,
        )

    def search_frames(
        self, pattern: str, *, project_id: str | None = "default", limit: int = 50
    ) -> list[dict]:
        return self._frames.search_frames(
            pattern,
            project_id=project_id,
            limit=limit,
        )

    # --- execution_log ---------------------------------------------------
    def log_cell(
        self,
        *,
        frame_id: str | None,
        code: str,
        result: dict,
        origin: str = "agent",
        cell_seq: int | None = None,
        project_id: str = "default",
        root_frame_id: str | None = None,
        cell_index: int | None = None,
        kernel_id: str = "python",
        language: str = "python",
        figures: list | None = None,
        files_read: list | None = None,
        files_written: list | None = None,
    ) -> str:
        return self._frames.log_cell(
            frame_id=frame_id,
            code=code,
            result=result,
            origin=origin,
            cell_seq=cell_seq,
            project_id=project_id,
            root_frame_id=root_frame_id,
            cell_index=cell_index,
            kernel_id=kernel_id,
            language=language,
            figures=figures,
            files_read=files_read,
            files_written=files_written,
        )

    def list_cells(self, root_frame_id: str) -> list[dict]:
        return self._frames.list_cells(root_frame_id)

    def cell_detail(self, producing_cell_id: str) -> dict | None:
        return self._frames.cell_detail(producing_cell_id)

    def delete_frame(self, frame_id: str) -> None:
        self._frames.delete_frame(frame_id)

    def get_frame(self, frame_id: str) -> dict | None:
        return self._frames.get_frame(frame_id)

    def get_artifact(self, artifact_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT a.*, v.size_bytes, v.checksum, v.path "
                "FROM artifacts a LEFT JOIN artifact_versions v "
                "ON a.latest_version_id=v.version_id WHERE a.artifact_id=?",
                (artifact_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_artifact(self, artifact_id: str) -> list[str]:
        """Remove an artifact + its versions. Returns the on-disk paths that are
        no longer referenced (caller may unlink them)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, snapshot_path FROM artifact_versions "
                "WHERE artifact_id=?",
                (artifact_id,),
            ).fetchall()
            # both the live path AND the immutable per-version snapshot are ours
            # to reclaim once the artifact is gone
            paths = {p for r in rows for p in (r["path"], r["snapshot_path"]) if p}
            self._conn.execute(
                "DELETE FROM artifact_versions WHERE artifact_id=?", (artifact_id,)
            )
            self._conn.execute(
                "DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,)
            )
            self._conn.execute(
                "DELETE FROM annotations WHERE artifact_id=?", (artifact_id,)
            )
            self._conn.commit()
            # keep any path a surviving version still references (as a live path or
            # a snapshot) — checked AFTER deletion so only OTHER artifacts count
            keep = set()
            for p in paths:
                if self._conn.execute(
                    "SELECT 1 FROM artifact_versions "
                    "WHERE path=? OR snapshot_path=? LIMIT 1",
                    (p, p),
                ).fetchone():
                    keep.add(p)
        return [p for p in paths if p not in keep]

    def rename_artifact(self, artifact_id: str, filename: str) -> None:
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "UPDATE artifacts SET filename=?, updated_at=? WHERE artifact_id=?",
                (filename, now, artifact_id),
            )
            self._conn.execute(
                "UPDATE artifact_versions SET filename=? WHERE artifact_id=?",
                (filename, artifact_id),
            )
            self._conn.commit()

    def artifact_by_filename(
        self, filename: str, root_frame_id: str | None = None, *, strict: bool = False
    ) -> dict | None:
        """Find an artifact by filename. With ``strict=True`` and a
        ``root_frame_id``, ONLY match within that session (no cross-session
        fallback) — used when versioning a re-written file so a common name like
        ``figure_cell1_1.png`` isn't mistaken for another session's artifact."""
        with self._lock:
            if root_frame_id:
                row = self._conn.execute(
                    "SELECT artifact_id FROM artifacts WHERE filename=? AND "
                    "root_frame_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                    (filename, root_frame_id),
                ).fetchone()
                if row:
                    return self.get_artifact(row["artifact_id"])
                if strict:
                    return None
            row = self._conn.execute(
                "SELECT artifact_id FROM artifacts WHERE filename=? "
                "ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (filename,),
            ).fetchone()
        return self.get_artifact(row["artifact_id"]) if row else None

    # --- artifacts -------------------------------------------------------
    def _artifact_write_scope(
        self,
        *,
        frame_id: str | None,
        root_frame_id: str | None,
        project_id: str | None,
    ) -> tuple[bool, str | None, str]:
        """Resolve and validate producer/root/project ownership for a write."""
        explicit_scope = any(
            value is not None for value in (frame_id, root_frame_id, project_id)
        )
        actor = self.get_frame(frame_id) if frame_id else None
        scope_source = frame_id if actor else (root_frame_id or frame_id)
        scope = self.resolve_frame_scope(
            scope_source,
            fallback_project=project_id or "default",
        )
        if actor:
            if root_frame_id is not None and root_frame_id != scope["root_frame_id"]:
                raise ValueError("root_frame_id conflicts with producer frame")
            if project_id is not None and project_id != scope["project_id"]:
                raise ValueError("project_id conflicts with producer frame")
            resolved_root = scope["root_frame_id"]
        else:
            resolved_root = root_frame_id or scope["root_frame_id"] or frame_id
        return explicit_scope, resolved_root, scope["project_id"]

    def save_artifact(
        self,
        *,
        path: str,
        filename: str,
        content_type: str | None,
        size_bytes: int,
        checksum: str | None,
        producing_cell_id: str | None = None,
        frame_id: str | None = None,
        root_frame_id: str | None = None,
        project_id: str | None = None,
        artifact_id: str | None = None,
        is_user_upload: bool = False,
        priority: int = 0,
        env_snapshot_id: str | None = None,
        snapshot_path: str | None = None,
    ) -> dict:
        """Register a new artifact version. ``path`` is the (live, possibly
        mutable) file; ``snapshot_path`` — when given — is an immutable per-version
        copy of the bytes so version history survives later in-place overwrites of
        ``path`` (see gateway._write_version_snapshot). The version row is written
        before the artifact row is (re)pointed at it, both under one commit, so
        ``latest_version_id`` never dangles."""
        explicit_scope, resolved_root, resolved_project = self._artifact_write_scope(
            frame_id=frame_id,
            root_frame_id=root_frame_id,
            project_id=project_id,
        )
        now = _now_ms()
        version_id = f"v-{uuid.uuid4().hex[:12]}"
        new_artifact = artifact_id is None
        if new_artifact:
            artifact_id = f"a-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if not new_artifact:
                current = self._conn.execute(
                    "SELECT project_id,root_frame_id FROM artifacts "
                    "WHERE artifact_id=?",
                    (artifact_id,),
                ).fetchone()
                if current is None:
                    raise KeyError(f"no such artifact {artifact_id!r}")
                if not explicit_scope:
                    resolved_root = current["root_frame_id"]
                    resolved_project = current["project_id"]
                if (
                    current["root_frame_id"] is not None
                    and resolved_root is not None
                    and current["root_frame_id"] != resolved_root
                ):
                    raise ValueError("artifact belongs to a different root frame")
                if (
                    current["root_frame_id"] is not None
                    and current["project_id"] != resolved_project
                ):
                    raise ValueError("artifact belongs to a different project")
            self._conn.execute(
                "INSERT INTO artifact_versions(version_id,artifact_id,filename,"
                "content_type,size_bytes,checksum,path,snapshot_path,"
                "producing_cell_id,frame_id,created_at,env_snapshot_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version_id,
                    artifact_id,
                    filename,
                    content_type,
                    size_bytes,
                    checksum,
                    path,
                    snapshot_path,
                    producing_cell_id,
                    frame_id,
                    now,
                    env_snapshot_id,
                ),
            )
            if new_artifact:
                self._conn.execute(
                    "INSERT INTO artifacts(artifact_id,project_id,root_frame_id,"
                    "filename,content_type,is_user_upload,priority,"
                    "latest_version_id,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        artifact_id,
                        resolved_project,
                        resolved_root,
                        filename,
                        content_type,
                        1 if is_user_upload else 0,
                        priority,
                        version_id,
                        now,
                        now,
                    ),
                )
            else:
                self._conn.execute(
                    "UPDATE artifacts SET latest_version_id=?,updated_at=? "
                    "WHERE artifact_id=?",
                    (version_id, now, artifact_id),
                )
            self._conn.commit()
        return {
            "artifact_id": artifact_id,
            "version_id": version_id,
            "filename": filename,
            "path": path,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "created_at": now,
        }

    def record_cell_artifact(
        self,
        *,
        path: str,
        filename: str,
        content_type: str | None,
        size_bytes: int,
        checksum: str | None,
        producing_cell_id: str | None,
        frame_id: str | None,
        root_frame_id: str | None = None,
        project_id: str | None = None,
        env_snapshot_id: str | None = None,
        snapshot_path: str | None = None,
        input_version_ids: list[str] | tuple[str, ...] | None = None,
        preserve_filename: bool = False,
        preserve_content_type: bool = False,
        reuse_policy: str = "any",
    ) -> dict:
        """Atomically record or finalize one cell's physical file write.

        Provenance reports arrive during kernel execution and capture enriches
        the same bytes after the cell.  When root, physical path, producing
        cell, and checksum all match an artifact's latest version, this method
        preserves that version id and fills missing capture metadata. Same-name
        candidates are preferred, but an explicit display filename does not
        split one physical cell output into two artifacts. The ``preserve_*``
        flags let automatic capture retain an earlier explicit label and MIME
        declaration, while still filling either field when it was absent. A
        different cell or different bytes always creates a new version.
        ``save_artifact`` keeps its unconditional-new semantics for uploads and
        explicit version creation. This transaction assumes the application's
        normal single Store writer; the process-local lock serializes callers
        sharing that Store instance. ``reuse_policy='provisional'`` is for an
        explicit in-cell save: it reuses an unsnapshotted provenance record but
        keeps repeated explicit saves as distinct versions.
        """
        if reuse_policy not in {"any", "provisional"}:
            raise ValueError(f"unknown cell artifact reuse policy: {reuse_policy!r}")
        _explicit, resolved_root, resolved_project = self._artifact_write_scope(
            frame_id=frame_id,
            root_frame_id=root_frame_id,
            project_id=project_id,
        )
        now = _now_ms()
        version_id: str
        artifact_id: str
        created_at = now
        stored_version: sqlite3.Row
        with self._lock:
            try:
                artifact = None
                candidate = None
                root_clause = (
                    "a.root_frame_id=?"
                    if resolved_root is not None
                    else "a.root_frame_id IS NULL"
                )
                root_args = (resolved_root,) if resolved_root is not None else ()

                # Search every scoped logical artifact for the exact latest
                # provenance version first.  A separate same-name artifact may
                # have been registered between the mid-cell provenance report
                # and post-cell capture; simply checking the newest artifact
                # would strand the lineage-bearing version in that race.
                if producing_cell_id and checksum is not None:
                    exact_rows = self._conn.execute(
                        "SELECT v.*,a.latest_version_id AS artifact_latest_version_id,"
                        "CASE WHEN a.filename=? THEN 0 ELSE 1 END AS filename_rank "
                        "FROM artifact_versions v JOIN artifacts a "
                        "ON a.artifact_id=v.artifact_id WHERE a.project_id=? AND "
                        + root_clause
                        + " AND v.producing_cell_id=? AND v.checksum=? "
                        "ORDER BY filename_rank,v.created_at DESC,v.rowid DESC",
                        (
                            filename,
                            resolved_project,
                            *root_args,
                            producing_cell_id,
                            checksum,
                        ),
                    ).fetchall()
                    for row in exact_rows:
                        if (
                            row["artifact_latest_version_id"] == row["version_id"]
                            and _same_file_path(row["path"], path)
                        ):
                            candidate = row
                            break

                reuse = candidate is not None and (
                    reuse_policy == "any" or not candidate["snapshot_path"]
                )

                if reuse:
                    artifact = self._conn.execute(
                        "SELECT rowid AS artifact_rowid,* FROM artifacts "
                        "WHERE artifact_id=?",
                        (candidate["artifact_id"],),
                    ).fetchone()
                else:
                    # A snapshotted exact candidate is an earlier explicit save,
                    # not a provisional record. Continue the requested logical
                    # filename if it exists; a different alias starts its own
                    # artifact instead of renaming the prior explicit result.
                    artifact = self._conn.execute(
                        "SELECT rowid AS artifact_rowid,* FROM artifacts a "
                        "WHERE a.filename=? AND a.project_id=? AND "
                        + root_clause
                        + " ORDER BY a.created_at DESC,a.rowid DESC LIMIT 1",
                        (filename, resolved_project, *root_args),
                    ).fetchone()

                if reuse:
                    artifact_id = candidate["artifact_id"]
                    version_id = candidate["version_id"]
                    created_at = candidate["created_at"]
                    stored_filename = (
                        (candidate["filename"] or artifact["filename"])
                        if preserve_filename
                        else filename
                    )
                    stored_content_type = (
                        candidate["content_type"]
                        if preserve_content_type and candidate["content_type"]
                        else content_type
                    )
                    self._conn.execute(
                        "UPDATE artifact_versions SET filename=?,"
                        "content_type=COALESCE(?,content_type),size_bytes=?,"
                        "checksum=?,path=?,snapshot_path=COALESCE(snapshot_path,?),"
                        "env_snapshot_id=COALESCE(env_snapshot_id,?) "
                        "WHERE version_id=?",
                        (
                            stored_filename,
                            stored_content_type,
                            size_bytes,
                            checksum,
                            path,
                            snapshot_path,
                            env_snapshot_id,
                            version_id,
                        ),
                    )
                else:
                    stored_filename = filename
                    stored_content_type = content_type
                    version_id = f"v-{uuid.uuid4().hex[:12]}"
                    artifact_id = (
                        artifact["artifact_id"]
                        if artifact is not None
                        else f"a-{uuid.uuid4().hex[:12]}"
                    )
                    self._conn.execute(
                        "INSERT INTO artifact_versions(version_id,artifact_id,"
                        "filename,content_type,size_bytes,checksum,path,"
                        "snapshot_path,producing_cell_id,frame_id,created_at,"
                        "env_snapshot_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            version_id,
                            artifact_id,
                            filename,
                            content_type,
                            size_bytes,
                            checksum,
                            path,
                            snapshot_path,
                            producing_cell_id,
                            frame_id,
                            now,
                            env_snapshot_id,
                        ),
                    )
                    if artifact is None:
                        self._conn.execute(
                            "INSERT INTO artifacts(artifact_id,project_id,"
                            "root_frame_id,filename,content_type,is_user_upload,"
                            "priority,latest_version_id,created_at,updated_at) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (
                                artifact_id,
                                resolved_project,
                                resolved_root,
                                filename,
                                stored_content_type,
                                0,
                                0,
                                version_id,
                                now,
                                now,
                            ),
                        )

                self._conn.execute(
                    "UPDATE artifacts SET filename=?,"
                    "content_type=COALESCE(?,content_type),latest_version_id=?,"
                    "updated_at=? WHERE artifact_id=?",
                    (
                        stored_filename,
                        stored_content_type,
                        version_id,
                        now,
                        artifact_id,
                    ),
                )
                seen_inputs: set[str] = set()
                for input_version_id in input_version_ids or ():
                    if (
                        not input_version_id
                        or input_version_id == version_id
                        or input_version_id in seen_inputs
                    ):
                        continue
                    seen_inputs.add(input_version_id)
                    exists = self._conn.execute(
                        "SELECT 1 FROM lineage_edges WHERE input_version_id=? "
                        "AND output_version_id=? LIMIT 1",
                        (input_version_id, version_id),
                    ).fetchone()
                    if exists:
                        continue
                    self._conn.execute(
                        "INSERT INTO lineage_edges(edge_id,input_version_id,"
                        "output_version_id,producing_cell_id,frame_id,created_at) "
                        "VALUES(?,?,?,?,?,?)",
                        (
                            f"e-{uuid.uuid4().hex[:12]}",
                            input_version_id,
                            version_id,
                            producing_cell_id,
                            frame_id,
                            now,
                        ),
                    )
                stored_version = self._conn.execute(
                    "SELECT * FROM artifact_versions WHERE version_id=?",
                    (version_id,),
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "artifact_id": artifact_id,
            "version_id": version_id,
            "filename": stored_version["filename"],
            "path": stored_version["path"],
            "content_type": stored_version["content_type"],
            "size_bytes": stored_version["size_bytes"],
            "checksum": stored_version["checksum"],
            "created_at": created_at,
        }

    # --- environment snapshots -------------------------------------------
    def upsert_env_snapshot(self, snapshot: dict) -> str:
        """Store a de-duplicated environment snapshot; return its snapshot_id.

        The id is a content hash of the interpreter identity + package manifest,
        so identical envs collapse to a single row (many figures share it)."""
        packages = snapshot.get("packages") or []
        pj = json.dumps(packages, separators=(",", ":"))
        remote = snapshot.get("remote") or []
        rj = json.dumps(remote, separators=(",", ":"), sort_keys=True)
        basis = "|".join(
            [
                snapshot.get("kind") or "",
                snapshot.get("python_version") or "",
                snapshot.get("implementation") or "",
                snapshot.get("platform") or "",
                pj,
                rj,  # remote-GPU job provenance makes a remotely-computed run distinct
            ]
        )
        sid = "env-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM env_snapshots WHERE snapshot_id=?", (sid,)
            ).fetchone()
            if not exists:
                self._conn.execute(
                    "INSERT INTO env_snapshots(snapshot_id,created_at,kind,"
                    "python_version,implementation,platform,package_count,"
                    "packages_json,remote_json) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        sid,
                        _now_ms(),
                        snapshot.get("kind"),
                        snapshot.get("python_version"),
                        snapshot.get("implementation"),
                        snapshot.get("platform"),
                        int(snapshot.get("package_count") or len(packages)),
                        pj,
                        rj if remote else None,
                    ),
                )
                self._conn.commit()
        return sid

    def get_env_snapshot(self, snapshot_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM env_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["packages"] = json.loads(d.pop("packages_json") or "[]")
        except (ValueError, TypeError):
            d.pop("packages_json", None)
            d["packages"] = []
        try:
            d["remote"] = json.loads(d.pop("remote_json") or "[]")
        except (ValueError, TypeError):
            d.pop("remote_json", None)
            d["remote"] = []
        return d

    def env_snapshot_for_artifact(
        self, artifact_id: str, version_id: str | None = None
    ) -> dict | None:
        """The env snapshot bound to a specific version, or the artifact's latest.

        Returns None when nothing was recorded (e.g. a user upload, or an artifact
        produced before this feature existed) — the caller falls back to live."""
        with self._lock:
            if version_id:
                row = self._conn.execute(
                    "SELECT env_snapshot_id FROM artifact_versions "
                    "WHERE version_id=? AND artifact_id=?",
                    (version_id, artifact_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT v.env_snapshot_id FROM artifacts a "
                    "JOIN artifact_versions v ON a.latest_version_id=v.version_id "
                    "WHERE a.artifact_id=?",
                    (artifact_id,),
                ).fetchone()
        sid = row["env_snapshot_id"] if row else None
        return self.get_env_snapshot(sid) if sid else None

    def list_artifacts(self, filters: dict | None = None) -> list[dict]:
        filters = filters or {}
        sql = (
            "SELECT a.artifact_id,a.filename,a.content_type,a.is_user_upload,"
            "a.priority,a.latest_version_id,a.root_frame_id,a.project_id,"
            "a.created_at,v.size_bytes,v.checksum "
            "FROM artifacts a LEFT JOIN artifact_versions v "
            "ON a.latest_version_id=v.version_id"
        )
        clauses, params = [], []
        for k in (
            "project_id",
            "content_type",
            "filename",
            "artifact_id",
            "root_frame_id",
        ):
            if k in filters:
                clauses.append(f"a.{k}=?")
                params.append(filters[k])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.created_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def resolve_artifact_path(self, ident: str) -> str | None:
        """On-disk file to serve for a version_id or artifact_id. Prefers the
        immutable per-version ``snapshot_path`` (so historical versions serve their
        OWN bytes, not the current live-file content), falling back to ``path``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(snapshot_path, path) AS p FROM artifact_versions "
                "WHERE version_id=?",
                (ident,),
            ).fetchone()
            if row:
                return row["p"]
            row = self._conn.execute(
                "SELECT COALESCE(v.snapshot_path, v.path) AS p FROM artifacts a "
                "JOIN artifact_versions v ON a.latest_version_id=v.version_id "
                "WHERE a.artifact_id=?",
                (ident,),
            ).fetchone()
        return row["p"] if row else None

    def version_for_path(self, path: str) -> str | None:
        """Reverse lookup the newest version for an exact or aliased path.

        The indexed lexical lookup gives us a lower bound. Newer rows are still
        checked for a physical alias before returning it, so an older exact row
        cannot hide a newer ``/tmp``/``/private/tmp`` or symlink spelling.
        Without an exact row, the physical fallback also preserves legacy
        relative-path records without rewriting history.
        """
        with self._lock:
            exact = self._conn.execute(
                "SELECT version_id,created_at,rowid AS version_rowid "
                "FROM artifact_versions WHERE path=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (str(path),),
            ).fetchone()
            identity = _file_identity(path)
            if identity is None:
                return exact["version_id"] if exact else None
            if exact:
                candidates = self._conn.execute(
                    "SELECT version_id,path FROM artifact_versions WHERE "
                    "created_at>? OR (created_at=? AND rowid>?) "
                    "ORDER BY created_at DESC, rowid DESC",
                    (
                        exact["created_at"],
                        exact["created_at"],
                        exact["version_rowid"],
                    ),
                ).fetchall()
            else:
                candidates = self._conn.execute(
                    "SELECT version_id,path FROM artifact_versions "
                    "ORDER BY created_at DESC, rowid DESC"
                ).fetchall()
        for candidate in candidates:
            if _file_identity(candidate["path"]) == identity:
                return candidate["version_id"]
        return exact["version_id"] if exact else None

    def version_meta(self, version_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifact_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_versions(self, artifact_id: str) -> list[dict]:
        """All versions of an artifact, newest first, each flagged is_latest."""
        with self._lock:
            latest = self._conn.execute(
                "SELECT latest_version_id FROM artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            rows = self._conn.execute(
                "SELECT version_id,filename,content_type,size_bytes,checksum,"
                "producing_cell_id,frame_id,created_at FROM artifact_versions "
                "WHERE artifact_id=? ORDER BY created_at DESC, rowid DESC",
                (artifact_id,),
            ).fetchall()
        lv = latest["latest_version_id"] if latest else None
        out = []
        for i, r in enumerate(rows):
            d = dict(r)
            d["is_latest"] = r["version_id"] == lv
            d["ordinal"] = len(rows) - i  # v1 = oldest
            out.append(d)
        return out

    def update_version_path(
        self,
        version_id: str,
        path: str,
        size_bytes: int | None = None,
        checksum: str | None = None,
    ) -> None:
        """Re-point a version at an (immutable) snapshot file so version history
        survives later in-place edits of the live workspace file."""
        sets = ["path=?"]
        params: list = [path]
        if size_bytes is not None:
            sets.append("size_bytes=?")
            params.append(size_bytes)
        if checksum is not None:
            sets.append("checksum=?")
            params.append(checksum)
        params.append(version_id)
        self._exec(
            f"UPDATE artifact_versions SET {','.join(sets)} " "WHERE version_id=?",
            tuple(params),
        )

    def set_version_snapshot(self, version_id: str, snapshot_path: str) -> None:
        """Bind a version to its immutable per-version byte snapshot, WITHOUT
        touching ``path`` (which stays the live workspace file so the provenance
        reverse-lookup ``version_for_path`` keeps resolving reads of that file)."""
        self._exec(
            "UPDATE artifact_versions SET snapshot_path=? " "WHERE version_id=?",
            (snapshot_path, version_id),
        )

    def set_priority(self, artifact_id: str, priority: int) -> dict | None:
        """priority > 0 = starred/pinned, < 0 = hidden, 0 = normal."""
        self._exec(
            "UPDATE artifacts SET priority=?,updated_at=? WHERE artifact_id=?",
            (int(priority), _now_ms(), artifact_id),
        )
        return self.get_artifact(artifact_id)

    def set_latest_version(self, artifact_id: str, version_id: str) -> dict | None:
        """Revert: make an existing version the current one. Validates the
        version belongs to the artifact. History is preserved."""
        with self._lock:
            row = self._conn.execute(
                "SELECT version_id FROM artifact_versions WHERE version_id=? "
                "AND artifact_id=?",
                (version_id, artifact_id),
            ).fetchone()
        if not row:
            return None
        self._exec(
            "UPDATE artifacts SET latest_version_id=?,updated_at=? "
            "WHERE artifact_id=?",
            (version_id, _now_ms(), artifact_id),
        )
        return self.get_artifact(artifact_id)

    # --- lineage ---------------------------------------------------------
    def add_lineage_edge(
        self,
        *,
        input_version_id: str,
        output_version_id: str,
        producing_cell_id: str | None = None,
        frame_id: str | None = None,
    ) -> None:
        self._exec(
            "INSERT INTO lineage_edges(edge_id,input_version_id,"
            "output_version_id,producing_cell_id,frame_id,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                f"e-{uuid.uuid4().hex[:12]}",
                input_version_id,
                output_version_id,
                producing_cell_id,
                frame_id,
                _now_ms(),
            ),
        )

    def lineage_inputs(self, version_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT le.input_version_id, av.filename, av.path "
                "FROM lineage_edges le LEFT JOIN artifact_versions av "
                "ON le.input_version_id=av.version_id "
                "WHERE le.output_version_id=?",
                (version_id,),
            ).fetchall()
        return [
            {
                "version_id": r["input_version_id"],
                "filename": r["filename"],
                "path": r["path"],
            }
            for r in rows
        ]

    def lineage_edges_for(self, version_id: str, direction: str) -> list[dict]:
        col_from = "output_version_id" if direction == "up" else "input_version_id"
        col_to = "input_version_id" if direction == "up" else "output_version_id"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {col_to} AS nxt FROM lineage_edges WHERE {col_from}=?",
                (version_id,),
            ).fetchall()
        return [r["nxt"] for r in rows]

    def producing_cell_for_version(self, version_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT el.code, el.frame_id, el.producing_cell_id "
                "FROM artifact_versions av "
                "LEFT JOIN execution_log el "
                "ON av.producing_cell_id=el.producing_cell_id "
                "WHERE av.version_id=?",
                (version_id,),
            ).fetchone()
        return dict(row) if row and row["code"] is not None else None

    # --- notes -----------------------------------------------------------
    def add_note(
        self, *, project_id: str, content: str, title: str | None = None
    ) -> dict:
        return self._notes.add(
            project_id=project_id,
            content=content,
            title=title,
        )

    def list_notes(self, project_id: str) -> list[dict]:
        return self._notes.list(project_id)

    def delete_note(self, note_id: str) -> None:
        self._notes.delete(note_id)

    # --- settings (KV) ---------------------------------------------------
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: str) -> None:
        self._settings.set(key, value)

    # --- model profiles (saved LLM/API configs) --------------------------
    # Stored as a JSON list under the `model_profiles` setting so users can keep
    # several full API configs (provider + base_url + model + key) side by side
    # and switch between them. Activating one writes the live `llm_*` settings.
    def list_model_profiles(self) -> list[dict]:
        return self._settings.list_model_profiles()

    def set_model_profiles(self, profiles: list[dict]) -> None:
        self._settings.set_model_profiles(profiles)

    def mutate_model_profiles(self, fn):
        return self._settings.mutate_model_profiles(fn)

    # --- permission rules (opencode-style tool-call gate) ----------------
    def set_permission_rule(
        self,
        *,
        scope: str,
        scope_id: str = "",
        tool: str,
        pattern: str = "*",
        decision: str,
    ) -> str:
        return self._permissions.set_rule(
            scope=scope,
            scope_id=scope_id,
            tool=tool,
            pattern=pattern,
            decision=decision,
        )

    def delete_permission_rule(self, rule_id: str) -> None:
        self._permissions.delete_rule(rule_id)

    def get_permission_rules(self, *, scope: str, scope_id: str = "") -> list[dict]:
        return self._permissions.get_rules(scope=scope, scope_id=scope_id)

    def list_permission_rules_for_frame(
        self, *, root_frame_id: str | None = None, project_id: str | None = None
    ) -> dict:
        return self._permissions.list_for_frame(
            root_frame_id=root_frame_id,
            project_id=project_id,
        )

    def resolve_permission(
        self,
        *,
        root_frame_id: str | None = None,
        project_id: str | None = None,
        tool: str,
        pattern_input: str = "",
    ) -> str:
        return self._permissions.resolve(
            root_frame_id=root_frame_id,
            project_id=project_id,
            tool=tool,
            pattern_input=pattern_input,
        )

    def seed_default_permission_rules(self, *, force: bool = False) -> None:
        self._permissions.seed_defaults(force=force)

    # --- plans (structured plan → approve → auto-execute) ----------------
    def _plan_row(self, row) -> dict:
        return self._plans.normalize_row(row)

    def create_plan(
        self,
        *,
        frame_id: str,
        project_id: str = "default",
        title: str | None,
        rationale: str | None,
        confidence: str | None,
        steps: list[dict],
        artifact_id: str | None = None,
        status: str = "draft",
    ) -> dict:
        return self._plans.create(
            frame_id=frame_id,
            project_id=project_id,
            title=title,
            rationale=rationale,
            confidence=confidence,
            steps=steps,
            artifact_id=artifact_id,
            status=status,
        )

    def get_plan(self, plan_id: str) -> dict | None:
        return self._plans.get(plan_id)

    def get_plan_by_frame(self, frame_id: str) -> dict | None:
        """The most recent (non-discarded) plan for a frame, else the newest."""
        return self._plans.get_by_frame(frame_id)

    def list_plans(self, frame_id: str, *, limit: int = 50) -> list[dict]:
        return self._plans.list_for_frame(frame_id, limit=limit)

    def update_plan(
        self,
        plan_id: str,
        *,
        title: str | None = None,
        rationale: str | None = None,
        confidence: str | None = None,
        steps: list[dict] | None = None,
        status: str | None = None,
        step_status: dict | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self._plans.update(
            plan_id,
            title=title,
            rationale=rationale,
            confidence=confidence,
            steps=steps,
            status=status,
            step_status=step_status,
            artifact_id=artifact_id,
        )

    def set_plan_step_status(
        self, plan_id: str, step_id: str, status: str, note: str | None = None
    ) -> dict | None:
        """Merge one step's status into the plan's step_status JSON. Returns the
        updated plan (with steps[] status folded in)."""
        return self._plans.set_step_status(plan_id, step_id, status, note)

    def delete_plans_for_frame(self, frame_id: str) -> None:
        self._plans.delete_for_frame(frame_id)

    # --- folders (session grouping within a project) --------------------
    def create_folder(self, *, project_id: str, name: str) -> dict:
        return self._folders.create(project_id=project_id, name=name)

    def list_folders(self, project_id: str) -> list[dict]:
        return self._folders.list(project_id)

    def rename_folder(self, folder_id: str, name: str) -> None:
        self._folders.rename(folder_id, name)

    def delete_folder(self, folder_id: str) -> None:
        self._folders.delete(folder_id)

    def set_frame_folder(self, frame_id: str, folder_id: str | None) -> None:
        self._folders.set_frame_folder(frame_id, folder_id)

    # --- memories --------------------------------------------------------
    def add_memory(
        self, *, content: str, block: str = "general", project_id: str = "default"
    ) -> dict:
        return self._memories.add(
            content=content,
            block=block,
            project_id=project_id,
        )

    def list_memories(
        self, project_id: str | None = None, block: str | None = None
    ) -> list[dict]:
        return self._memories.list(project_id=project_id, block=block)

    def delete_memory(self, memory_id: str) -> None:
        self._memories.delete(memory_id)

    def memory_blocks(self, project_id: str | None = None) -> list[dict]:
        return self._memories.blocks(project_id)

    # --- feedback (per message) -----------------------------------------
    def set_feedback(self, frame_id: str, key: str, rating: str | None) -> None:
        self._settings.set_feedback(frame_id, key, rating)

    def list_feedback(self, frame_id: str) -> dict:
        return self._settings.list_feedback(frame_id)

    # --- image annotations (figure review) ------------------------------
    def add_annotation(
        self,
        *,
        root_frame_id: str,
        artifact_id: str,
        artifact_name: str | None,
        rel_x: float,
        rel_y: float,
        body: str,
    ) -> dict:
        return self._annotations.add(
            root_frame_id=root_frame_id,
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            rel_x=rel_x,
            rel_y=rel_y,
            body=body,
        )

    def get_annotation(self, annotation_id: str) -> dict | None:
        return self._annotations.get(annotation_id)

    def list_annotations(
        self,
        root_frame_id: str,
        *,
        artifact_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        return self._annotations.list_for_frame(
            root_frame_id,
            artifact_id=artifact_id,
            status=status,
        )

    def update_annotation(
        self, annotation_id: str, *, body: str | None = None, status: str | None = None
    ) -> dict | None:
        return self._annotations.update(
            annotation_id,
            body=body,
            status=status,
        )

    def mark_annotations_sent(self, annotation_ids: list[str]) -> None:
        self._annotations.mark_sent(annotation_ids)

    def delete_annotation(self, annotation_id: str) -> None:
        self._annotations.delete(annotation_id)

    # --- global search (command palette) --------------------------------
    def search(self, query: str, limit: int = 20) -> dict:
        """Search sessions (name/task_summary) + artifacts (filename) for the
        ⌘K command palette."""
        q = f"%{query.strip()}%"
        with self._lock:
            frames = self._conn.execute(
                "SELECT frame_id,project_id,name,task_summary,updated_at FROM frames "
                "WHERE parent_id IS NULL AND (name LIKE ? OR task_summary LIKE ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (q, q, limit),
            ).fetchall()
            arts = self._conn.execute(
                "SELECT artifact_id,filename,content_type,root_frame_id,project_id "
                "FROM artifacts WHERE filename LIKE ? ORDER BY created_at DESC "
                "LIMIT ?",
                (q, limit),
            ).fetchall()
        return {
            "sessions": [
                {
                    "id": r["frame_id"],
                    "project_id": r["project_id"],
                    "name": r["name"],
                    "task_summary": r["task_summary"],
                }
                for r in frames
            ],
            "artifacts": [
                {
                    "id": r["artifact_id"],
                    "filename": r["filename"],
                    "content_type": r["content_type"],
                    "root_frame_id": r["root_frame_id"],
                    "project_id": r["project_id"],
                }
                for r in arts
            ],
        }

    # --- agents / specialists -------------------------------------------
    def list_agents(self) -> list[dict]:
        return self._agents.list()

    def get_agent(self, name: str) -> dict | None:
        return self._agents.get(name)

    def upsert_agent(
        self,
        *,
        name: str,
        description: str = "",
        system_prompt: str = "",
        skill_names: list | None = None,
        connectors: list | None = None,
        unrestricted: bool = True,
    ) -> dict:
        return self._agents.upsert(
            name=name,
            description=description,
            system_prompt=system_prompt,
            skill_names=skill_names,
            connectors=connectors,
            unrestricted=unrestricted,
        )

    def delete_agent(self, name: str) -> None:
        self._agents.delete(name)

    # --- connectors (MCP servers) ---------------------------------------
    def list_connectors(self) -> list[dict]:
        return self._connectors.list()

    def get_connector(self, connector_id: str) -> dict | None:
        return self._connectors.get(connector_id)

    def upsert_connector(
        self,
        *,
        connector_id: str,
        name: str,
        command,
        description: str = "",
        args=None,
        env=None,
        enabled: bool = True,
    ) -> dict:
        return self._connectors.upsert(
            connector_id=connector_id,
            name=name,
            command=command,
            description=description,
            args=args,
            env=env,
            enabled=enabled,
        )

    def set_connector_enabled(self, connector_id: str, enabled: bool) -> None:
        self._connectors.set_enabled(connector_id, enabled)

    def delete_connector(self, connector_id: str) -> None:
        self._connectors.delete(connector_id)

    # --- compaction ------------------------------------------------------
    def archive_compaction(
        self,
        *,
        frame_id: str | None,
        summary: str,
        compacted: list[dict],
        project_id: str = "default",
    ) -> str:
        return self._compactions.archive(
            frame_id=frame_id,
            summary=summary,
            compacted=compacted,
            project_id=project_id,
        )

    # --- endpoints ----------------------------------------------
    def upsert_endpoint(self, name: str, **fields: Any) -> None:
        self._endpoints.upsert(name, **fields)

    def list_endpoints(self) -> list[dict]:
        return self._endpoints.list()

    # --- host_call audit ----------------------------------------
    def log_host_call(
        self, *, method: str, args: list, ok: bool, frame_id: str | None = None
    ) -> None:
        self._host_calls.log(
            method=method,
            args=args,
            ok=ok,
            frame_id=frame_id,
        )

    # --- generic read-only query (host.query backing) -------------------
    def query(
        self,
        sql: str,
        params: list | None = None,
        limit: int | None = None,
        timeout_s: float = 5.0,
    ) -> list[dict]:
        """Run a read-only SELECT/CTE. Enforces denylist + timeout."""
        lowered = sql.lower()
        # Denylist check runs against a literal-stripped copy so a denied name
        # inside a string literal/comment is not a false positive, while a real
        # (possibly identifier-quoted) table reference still trips it.
        deny_scan = _strip_sql_literals(lowered)
        for bad in QUERY_DENYLIST:
            if bad in deny_scan:
                raise PermissionError(f"host.query: table '{bad}' is not readable")
        stripped = lowered.lstrip()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("host.query only allows read-only SELECT/CTE")
        for kw in (
            " insert ",
            " update ",
            " delete ",
            " drop ",
            " alter ",
            " create ",
            " attach ",
            " pragma ",
        ):
            if kw in f" {lowered} ":
                raise ValueError(f"host.query: forbidden keyword {kw.strip()!r}")
        # per-statement timeout via a busy interrupt handler
        with self._lock:
            self._conn.set_progress_handler(_TimeoutGuard(timeout_s), 10000)
            try:
                cur = self._conn.execute(sql, tuple(params or ()))
                rows = cur.fetchmany(limit) if limit else cur.fetchall()
            finally:
                self._conn.set_progress_handler(None, 10000)
        return [dict(r) for r in rows]

    def schema(self) -> dict[str, list[str]]:
        with self._lock:
            tables = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            out: dict[str, list[str]] = {}
            for t in tables:
                name = t["name"]
                if name in QUERY_DENYLIST or name.startswith("sqlite_"):
                    continue
                cols = self._conn.execute(f"PRAGMA table_info({name})").fetchall()
                out[name] = [c["name"] for c in cols]
        return out


class _TimeoutGuard:
    """Progress-handler callback that aborts a query after timeout_s (5s)."""

    def __init__(self, timeout_s: float):
        self._deadline = time.time() + timeout_s

    def __call__(self) -> int:
        return 1 if time.time() > self._deadline else 0


_STORES: dict[str, Store] = {}
_STORES_LOCK = threading.Lock()


def get_store(db_path: Path) -> Store:
    """Process-wide singleton Store per db path."""
    key = str(Path(db_path).resolve())
    with _STORES_LOCK:
        st = _STORES.get(key)
        if st is None:
            st = Store(Path(db_path))
            _STORES[key] = st
    return st
