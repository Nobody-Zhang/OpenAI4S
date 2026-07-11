"""Session-local credential behavior for host RPC calls."""

from __future__ import annotations


class CredentialService:
    """Keep short-lived credentials in memory for one dispatcher session.

    This service intentionally has no persistence, logging, permission, or
    replay behavior.  Those security policies remain in ``HostDispatcher`` so
    the raw value never leaves the narrow set/get RPC response path.
    """

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def set(self, spec: dict) -> dict:
        name = spec["name"]
        self._values[name] = spec.get("value", "")
        return {"ok": True, "name": name}

    def get(self, name: str) -> dict:
        if name not in self._values:
            raise KeyError(f"no credential {name!r}")
        return {"name": name, "value": self._values[name]}

    def list(self) -> list[str]:
        """Return stable credential names without exposing any values."""
        return sorted(self._values)


__all__ = ["CredentialService"]
