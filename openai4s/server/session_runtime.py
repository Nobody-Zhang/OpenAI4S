"""Session-scoped control-plane runtime, independent of language workers.

The dispatcher belongs to the Web session.  Python and R kernels are optional
execution-plane resources that may be started, replaced, or stopped without
discarding control-plane state such as approvals, completion, and delegation.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class SessionRuntime:
    """Own one lazily constructed dispatcher for the lifetime of a session."""

    def __init__(self, dispatcher: Any = None) -> None:
        self._dispatcher = dispatcher
        self._lock = threading.RLock()

    @property
    def dispatcher(self) -> Any:
        with self._lock:
            return self._dispatcher

    @dispatcher.setter
    def dispatcher(self, value: Any) -> None:
        self.bind(value)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._dispatcher is not None

    def bind(self, dispatcher: Any) -> Any:
        """Install a dispatcher explicitly (also preserves test compatibility)."""
        with self._lock:
            self._dispatcher = dispatcher
            return dispatcher

    def ensure(self, factory: Callable[[], Any]) -> Any:
        """Return the existing dispatcher or construct it exactly once."""
        with self._lock:
            if self._dispatcher is None:
                dispatcher = factory()
                if dispatcher is None:
                    raise RuntimeError("session dispatcher factory returned None")
                self._dispatcher = dispatcher
            return self._dispatcher


__all__ = ["SessionRuntime"]
