"""Workspace-confined file operations for host RPC calls."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Callable

_SECRET_BASENAMES = (
    "*.env",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".pgpass",
)


def is_secret_path(path: str) -> bool:
    """Return whether a basename belongs to the host tool secret denylist."""
    import posixpath

    basename = posixpath.basename((path or "").replace("\\", "/").rstrip("/")).lower()
    if not basename:
        return False
    return any(fnmatch.fnmatchcase(basename, pattern) for pattern in _SECRET_BASENAMES)


class WorkspaceFileService:
    """Execute file tools inside the workspace for the current frame.

    ``frame_id`` is a provider rather than a captured value because the CLI may
    assign its root frame after constructing the dispatcher.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        frame_id: Callable[[], str | None],
    ) -> None:
        self._data_dir = data_dir
        self._frame_id = frame_id

    def workspace(self) -> Path:
        """Return the resolved workspace, creating it on first use."""
        workspace = (
            self._data_dir
            / "agent-workspaces"
            / (self._frame_id() or "default")
        ).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def relative(self, path: Path) -> str | None:
        """Return a confined workspace-relative path, or ``None`` on escape."""
        try:
            return str(path.resolve().relative_to(self.workspace()))
        except (ValueError, OSError):
            return None

    def resolve(self, relative: str, *, must_exist: bool = False) -> Path:
        """Resolve a path and reject parent, absolute, and symlink escapes."""
        workspace = self.workspace()
        path = Path(relative)
        target = (path if path.is_absolute() else workspace / path).resolve()
        try:
            target.relative_to(workspace)
        except ValueError:
            raise ValueError(
                f"path escapes the workspace: {relative!r} "
                "(stay inside your working dir)"
            )
        if must_exist and not target.exists():
            raise FileNotFoundError(f"no such file: {relative}")
        return target

    def read_file(self, spec: dict) -> dict:
        path = self.resolve(spec.get("path", ""), must_exist=True)
        offset = max(0, int(spec.get("offset") or 0))
        limit = max(1, int(spec.get("limit") or 2000))
        try:
            data = path.read_bytes()
        except OSError as error:
            return {"error": f"read_file: {error}"}
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "path": self.relative(path),
                "binary": True,
                "size_bytes": len(data),
                "content": "",
            }
        lines = content.splitlines()
        window = lines[offset : offset + limit]
        return {
            "path": self.relative(path),
            "total_lines": len(lines),
            "offset": offset,
            "content": "\n".join(window),
            "truncated": (offset + limit) < len(lines),
        }

    def write_file(self, spec: dict) -> dict:
        path = self.resolve(spec.get("path", ""))
        content = spec.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": self.relative(path),
            "bytes": len(content.encode("utf-8")),
        }

    def edit_file(self, spec: dict) -> dict:
        path = self.resolve(spec.get("path", ""), must_exist=True)
        old = spec.get("old_string", "")
        new = spec.get("new_string", "")
        replace_all = bool(spec.get("replace_all"))
        content = path.read_text(encoding="utf-8")
        matches = content.count(old)
        if not old or matches == 0:
            return {"error": "edit_file: old_string not found"}
        if matches > 1 and not replace_all:
            return {
                "error": f"edit_file: old_string is not unique ({matches} matches); "
                "pass replace_all=True or add more context"
            }
        content = (
            content.replace(old, new)
            if replace_all
            else content.replace(old, new, 1)
        )
        path.write_text(content, encoding="utf-8")
        return {"path": self.relative(path), "replaced": matches}

    def glob(self, spec: dict) -> dict:
        pattern = spec.get("pattern") or "**/*"
        base = (
            self.resolve(spec.get("path"))
            if spec.get("path")
            else self.workspace()
        )
        matches = []
        for path in sorted(base.glob(pattern)):
            relative = self.relative(path) if path.is_file() else None
            if relative is not None and not is_secret_path(relative):
                matches.append(relative)
        return {
            "pattern": pattern,
            "count": len(matches),
            "matches": matches[:1000],
        }

    def grep(self, spec: dict) -> dict:
        pattern = spec.get("pattern") or ""
        if not pattern:
            return {"error": "grep: empty pattern"}
        try:
            regex = re.compile(pattern)
        except re.error as error:
            return {"error": f"grep: bad regex: {error}"}
        include = spec.get("include")
        base = (
            self.resolve(spec.get("path"))
            if spec.get("path")
            else self.workspace()
        )
        hits: list[dict] = []
        paths = base.glob(include) if include else base.rglob("*")
        for path in sorted(paths):
            if not path.is_file():
                continue
            relative = self.relative(path)
            if relative is None or is_secret_path(relative):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    hits.append(
                        {"file": relative, "line": line_number, "text": line[:400]}
                    )
                    if len(hits) >= 200:
                        return {
                            "pattern": pattern,
                            "count": len(hits),
                            "matches": hits,
                            "truncated": True,
                        }
        return {"pattern": pattern, "count": len(hits), "matches": hits}

    def list_dir(self, spec: dict) -> dict:
        relative = spec.get("path") or "."
        base = self.resolve(relative) if relative != "." else self.workspace()
        if not base.exists():
            return {"error": f"list_dir: no such directory: {relative}"}
        entries = []
        for path in sorted(base.iterdir()):
            entries.append(
                {
                    "name": path.name,
                    "path": self.relative(path) or path.name,
                    "is_dir": path.is_dir(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                }
            )
        return {"path": relative, "count": len(entries), "entries": entries}


__all__ = ["WorkspaceFileService", "is_secret_path"]
