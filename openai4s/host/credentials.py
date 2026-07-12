"""Session-local credential behavior for host RPC calls."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class _CredentialLease:
    name: str
    value: str
    purpose: str
    binding: str
    expires_at: float


class CredentialService:
    """Keep short-lived credentials in memory for one dispatcher session.

    This service intentionally has no persistence, logging, permission, or
    replay behavior.  Those security policies remain in ``HostDispatcher`` so
    the raw value never leaves the narrow set/get RPC response path.
    """

    def __init__(self, *, clock=time.monotonic) -> None:
        self._values: dict[str, str] = {}
        self._leases: dict[str, _CredentialLease] = {}
        self._lock = threading.RLock()
        self._clock = clock

    def set(self, spec: dict) -> dict:
        name = spec["name"]
        with self._lock:
            self._values[name] = spec.get("value", "")
            # Overwriting a credential invalidates every outstanding lease for
            # the old value; no stale capability may redeem after rotation.
            self._leases = {
                token: lease
                for token, lease in self._leases.items()
                if lease.name != name
            }
        return {"ok": True, "name": name}

    def issue(
        self,
        name: str,
        *,
        purpose: str = "host credential access",
        binding: str = "",
        ttl_seconds: float = 30.0,
    ) -> dict:
        """Mint one opaque, action-bound, single-use secret capability."""

        ttl = float(ttl_seconds)
        if not 0 < ttl <= 300:
            raise ValueError("credential lease ttl must be in (0, 300] seconds")
        with self._lock:
            self._purge_expired_locked()
            if name not in self._values:
                raise KeyError(f"no credential {name!r}")
            token = "cred-" + secrets.token_urlsafe(24)
            lease = _CredentialLease(
                name=name,
                value=self._values[name],
                purpose=str(purpose or "host credential access")[:200],
                binding=str(binding or "")[:500],
                expires_at=self._clock() + ttl,
            )
            self._leases[token] = lease
        return {
            "token": token,
            "name": name,
            "purpose": lease.purpose,
            "expires_in": ttl,
            "single_use": True,
        }

    def redeem(self, token: str, *, binding: str = "") -> dict:
        """Atomically consume one lease before releasing its value."""

        with self._lock:
            self._purge_expired_locked()
            # Pop first: even a later binding error cannot leave a reusable
            # token behind for probing/replay.
            lease = self._leases.pop(str(token or ""), None)
            if lease is None:
                raise KeyError("credential lease is unknown, expired, or consumed")
            if lease.binding != str(binding or "")[:500]:
                raise PermissionError("credential lease belongs to another action")
        return {"name": lease.name, "value": lease.value, "purpose": lease.purpose}

    def get(self, name: str, *, binding: str = "") -> dict:
        """Compatibility read implemented as an immediate one-shot lease."""

        issued = self.issue(name, binding=binding)
        redeemed = self.redeem(issued["token"], binding=binding)
        return {"name": redeemed["name"], "value": redeemed["value"]}

    def list(self) -> list[str]:
        """Return stable credential names without exposing any values."""
        with self._lock:
            return sorted(self._values)

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        self._leases = {
            token: lease
            for token, lease in self._leases.items()
            if lease.expires_at > now
        }


__all__ = ["CredentialService"]
