"""Narrow runtime ports available to concrete control tools.

Tools depend on these structural protocols instead of the large
``HostDispatcher``. The dispatcher remains the policy envelope and supplies
objects that implement the relevant port only after a call has been approved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class WorkspaceToolContext(Protocol):
    """Workspace path boundary used by file and search tools."""

    def workspace(self) -> Path: ...

    def relative(self, path: Path) -> str | None: ...

    def resolve(self, relative: str, *, must_exist: bool = False) -> Path: ...

    def is_secret_path(self, path: str) -> bool: ...


class EnvironmentToolContext(Protocol):
    """Mutable session hooks required by environment control tools."""

    active_env_bin: str | None
    active_r_env: str | None
    on_env_switch: Callable[[str], None] | None


__all__ = ["WorkspaceToolContext", "EnvironmentToolContext"]
