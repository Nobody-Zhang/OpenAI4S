"""Build-first, replay-safe, verifiable scientific Kernel recovery.

This module is deliberately protocol-neutral.  A candidate may be the existing
Python manager, the R sibling, or a future Jupyter adapter; callers inject the
small build/bootstrap/execute/inspect/publish callbacks.  The old generation is
never stopped or replaced here.  ``publish`` is called exactly once and only
after every required validation succeeds.

Recovery recipes contain code, but an explicit ``safe`` label is not trusted by
itself.  Python AST and conservative language-neutral scans reject completion,
shell, credentials, external writes, background/delegated work, remote jobs,
and unknown Host calls.  Rejected/non-replayable steps produce ``partial``—the
system never claims that an arbitrary in-memory namespace survived.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

REPLAY_SAFE = "safe"
REPLAY_CONDITIONAL = "conditional"
REPLAY_NEVER = "never"
REPLAY_POLICIES = frozenset({REPLAY_SAFE, REPLAY_CONDITIONAL, REPLAY_NEVER})

_UNSAFE_HOST_METHODS = frozenset(
    {
        "submit_output",
        "bash",
        "credentials_set",
        "credentials_get",
        "exec_background",
        "exec_interrupt",
        "delegate",
        "stop_child",
        "send_message",
        "compute_submit",
        "compute_cancel",
        "compute_close",
        "fold",
        "score_mutations",
        "mcp_call",
        "write_file",
        "edit_file",
        "save_artifact",
        "request_network_access",
        "endpoints_register",
        "skills_edit",
        "skills_publish",
        "skills_delete",
        "env_setup",
    }
)
_SAFE_HOST_METHODS = frozenset(
    {
        "capabilities",
        "current_model",
        "list_models",
        "query",
        "query_schema",
        "artifacts",
        "artifact_path",
        "frames",
        "lineage_get",
        "lineage_graph",
        "skills_list",
        "skills_get",
        "skills_read",
        "search_skills",
        "env_list",
        "todo_read",
        "plan_read",
    }
)
_RISKY_TEXT = re.compile(
    r"(?i)\b(subprocess|os\.system|system2?|shell|requests?\.|urllib\.|"
    r"socket\.|curl\b|wget\b|ssh\b|scp\b|writeLines\s*\(|saveRDS\s*\()"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SidecarManifest:
    """Exact bytes of one sidecar that was actually loaded."""

    name: str
    source: bytes
    order: int
    exports: tuple[str, ...] = ()
    import_mode: str = "module"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("sidecar name must be non-empty")
        if isinstance(self.order, bool) or self.order < 0:
            raise ValueError("sidecar order must be non-negative")
        if not isinstance(self.source, bytes):
            raise TypeError("sidecar source must be bytes")
        try:
            compile(self.source, self.source_path or f"<{self.name}>", "exec")
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"sidecar {self.name!r} does not compile: {error}") from error

    @property
    def sha256(self) -> str:
        return _sha256(self.source)

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "source_b64": base64.b64encode(self.source).decode("ascii"),
            "order": self.order,
            "exports": list(self.exports),
            "import_mode": self.import_mode,
            "source_path": self.source_path,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "SidecarManifest":
        try:
            source = base64.b64decode(str(value["source_b64"]), validate=True)
        except Exception as error:  # noqa: BLE001 — imported manifests are untrusted
            raise ValueError("invalid sidecar source encoding") from error
        sidecar = cls(
            name=str(value.get("name") or ""),
            source=source,
            order=int(value.get("order") or 0),
            exports=tuple(str(item) for item in (value.get("exports") or ())),
            import_mode=str(value.get("import_mode") or "module"),
            source_path=(
                str(value["source_path"]) if value.get("source_path") else None
            ),
        )
        if value.get("sha256") != sidecar.sha256:
            raise ValueError(f"sidecar hash mismatch: {sidecar.name}")
        return sidecar


@dataclass(frozen=True)
class BootstrapManifest:
    language: str
    interpreter: str
    runtime_version: str
    working_directory: str
    environment: Mapping[str, Any] = field(default_factory=dict)
    sdk_version: str | None = None
    provenance_version: str | None = None
    random_seed_policy: str | None = None
    sidecars: tuple[SidecarManifest, ...] = ()
    init_hooks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.language not in {"python", "r"}:
            raise ValueError("bootstrap language must be python or r")
        if not self.interpreter or not self.working_directory:
            raise ValueError("bootstrap interpreter and working_directory are required")
        orders = [sidecar.order for sidecar in self.sidecars]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("sidecar load order must be unique and sorted")

    def record(self) -> dict[str, Any]:
        return {
            "version": 1,
            "language": self.language,
            "interpreter": self.interpreter,
            "runtime_version": self.runtime_version,
            "working_directory": self.working_directory,
            "environment": dict(self.environment),
            "sdk_version": self.sdk_version,
            "provenance_version": self.provenance_version,
            "random_seed_policy": self.random_seed_policy,
            "sidecars": [sidecar.record() for sidecar in self.sidecars],
            "init_hooks": list(self.init_hooks),
        }

    @property
    def manifest_id(self) -> str:
        return "boot-" + _sha256(_canonical_bytes(self.record()))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "BootstrapManifest":
        if int(value.get("version") or 0) != 1:
            raise ValueError("unsupported bootstrap manifest version")
        sidecars = tuple(
            SidecarManifest.from_record(item)
            for item in (value.get("sidecars") or ())
            if isinstance(item, Mapping)
        )
        return cls(
            language=str(value.get("language") or ""),
            interpreter=str(value.get("interpreter") or ""),
            runtime_version=str(value.get("runtime_version") or ""),
            working_directory=str(value.get("working_directory") or ""),
            environment=(
                dict(value["environment"])
                if isinstance(value.get("environment"), Mapping)
                else {}
            ),
            sdk_version=(str(value["sdk_version"]) if value.get("sdk_version") else None),
            provenance_version=(
                str(value["provenance_version"])
                if value.get("provenance_version")
                else None
            ),
            random_seed_policy=(
                str(value["random_seed_policy"])
                if value.get("random_seed_policy")
                else None
            ),
            sidecars=sidecars,
            init_hooks=tuple(str(item) for item in (value.get("init_hooks") or ())),
        )


@dataclass(frozen=True)
class RecoveryStep:
    kind: str
    payload: Mapping[str, Any]
    replay_policy: str = REPLAY_NEVER
    step_id: str = field(default_factory=lambda: f"rs-{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        if self.replay_policy not in REPLAY_POLICIES:
            raise ValueError(f"unknown replay policy: {self.replay_policy}")
        if not self.kind.strip():
            raise ValueError("recovery step kind must be non-empty")


@dataclass(frozen=True)
class RecoveryRecipe:
    steps: tuple[RecoveryStep, ...] = ()
    required_symbols: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    environment_requirements: Mapping[str, Any] = field(default_factory=dict)
    namespace_coverage: str = "empty"

    def __post_init__(self) -> None:
        if self.namespace_coverage not in {"empty", "verified", "unverified"}:
            raise ValueError(
                f"unknown namespace coverage: {self.namespace_coverage!r}"
            )


@dataclass(frozen=True)
class RecoveryResult:
    recovery_id: str
    status: str
    source_generation_id: str | None
    candidate_generation_id: str | None
    manifest_id: str
    issues: tuple[dict[str, Any], ...] = ()
    replayed_steps: tuple[str, ...] = ()
    skipped_steps: tuple[str, ...] = ()


class Candidate(Protocol):
    generation_id: str

    def shutdown(self) -> None: ...


JournalSink = Callable[[dict[str, Any]], Any]


class RecoveryCancelled(RuntimeError):
    """Raised internally when the exact recovery ticket is cancelled."""


def replay_safety_error(
    code: str,
    *,
    language: str,
    declared_host_methods: Iterable[str] = (),
) -> str | None:
    """Return why a cell cannot be replayed, or ``None`` when admissible."""

    if not isinstance(code, str) or not code.strip():
        return "empty recovery cell"
    methods = {str(item) for item in declared_host_methods}
    unsafe = sorted(methods & _UNSAFE_HOST_METHODS)
    if unsafe:
        return "unsafe Host methods: " + ", ".join(unsafe)
    unknown = sorted(methods - _UNSAFE_HOST_METHODS - _SAFE_HOST_METHODS)
    if unknown:
        return "unknown Host methods: " + ", ".join(unknown)
    if _RISKY_TEXT.search(code):
        return "cell contains direct process, filesystem, or network execution"
    if language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return f"cell does not parse: {error}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
                if any(
                    root
                    in {
                        "os",
                        "pathlib",
                        "shutil",
                        "subprocess",
                        "socket",
                        "requests",
                        "urllib",
                    }
                    for root in roots
                ):
                    return "cell imports a direct process/filesystem/network module"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"open", "exec", "eval", "compile"}:
                    return f"recovery cell uses unsafe builtin: {node.func.id}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id == "host":
                    method = node.func.attr
                    if method in _UNSAFE_HOST_METHODS:
                        return f"unsafe Host method: {method}"
                    if method not in _SAFE_HOST_METHODS:
                        return f"unknown Host method: {method}"
                if node.func.attr in {
                    "mkdir",
                    "rename",
                    "replace",
                    "rmdir",
                    "save",
                    "savefig",
                    "to_csv",
                    "to_excel",
                    "to_json",
                    "to_parquet",
                    "to_pickle",
                    "unlink",
                    "write",
                    "write_bytes",
                    "write_text",
                }:
                    return f"recovery cell may write external state: {node.func.attr}"
    else:
        lowered = code.lower()
        if any(
            marker in lowered
            for marker in ("host$", "submit_output", "system(", "system2(")
        ):
            return "R recovery cell contains Host/shell side effects"
    return None


class KernelRecoveryOrchestrator:
    """Run one recovery transaction and atomically publish only verified state."""

    def __init__(
        self,
        *,
        build_candidate: Callable[[BootstrapManifest], Candidate],
        bootstrap_candidate: Callable[[Candidate, BootstrapManifest], Any],
        hydrate_workspace: Callable[[Candidate, Mapping[str, Any]], Any],
        hydrate_artifact: Callable[[Candidate, Mapping[str, Any]], Any],
        execute_cell: Callable[[Candidate, str, str], Mapping[str, Any]],
        inspect_symbols: Callable[[Candidate, str], Iterable[str]],
        artifact_digest: Callable[[Candidate, str], str | None],
        inspect_environment: Callable[[Candidate], Mapping[str, Any]],
        publish: Callable[[Candidate], Any],
        journal: JournalSink | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._build = build_candidate
        self._bootstrap = bootstrap_candidate
        self._hydrate_workspace = hydrate_workspace
        self._hydrate_artifact = hydrate_artifact
        self._execute = execute_cell
        self._symbols = inspect_symbols
        self._artifact_digest = artifact_digest
        self._environment = inspect_environment
        self._publish = publish
        self._journal = journal or (lambda _event: None)
        self._cancelled = cancelled or (lambda: False)

    def restore(
        self,
        *,
        root_frame_id: str,
        branch_id: str | None,
        manifest: BootstrapManifest,
        recipe: RecoveryRecipe,
        source_generation_id: str | None,
        recovery_id: str | None = None,
    ) -> RecoveryResult:
        recovery_id = recovery_id or f"recovery-{uuid.uuid4().hex[:16]}"
        branch_id = branch_id or root_frame_id
        candidate: Candidate | None = None
        candidate_id: str | None = None
        replayed: list[str] = []
        skipped: list[str] = []
        issues: list[dict[str, Any]] = []
        published = False

        def check_cancelled() -> None:
            if self._cancelled():
                raise RecoveryCancelled("recovery execution was cancelled")

        def record(phase: str, status: str, detail: Any = None) -> None:
            event = {
                "recovery_id": recovery_id,
                "root_frame_id": root_frame_id,
                "branch_id": branch_id,
                "source_generation_id": source_generation_id,
                "candidate_generation_id": candidate_id,
                "phase": phase,
                "status": status,
                "detail": detail if detail is not None else {},
            }
            self._journal(event)

        record("restore", "started", {"manifest_id": manifest.manifest_id})
        try:
            check_cancelled()
            candidate = self._build(manifest)
            candidate_id = str(candidate.generation_id)
            record("build", "completed")
            check_cancelled()
            self._bootstrap(candidate, manifest)
            record(
                "bootstrap",
                "completed",
                {
                    "manifest_id": manifest.manifest_id,
                    "sidecar_hashes": {
                        sidecar.name: sidecar.sha256 for sidecar in manifest.sidecars
                    },
                },
            )

            for step in recipe.steps:
                check_cancelled()
                if step.kind == "hydrate_workspace":
                    self._hydrate_workspace(candidate, step.payload)
                    record(step.kind, "completed", {"step_id": step.step_id})
                    continue
                if step.kind == "hydrate_artifact":
                    self._hydrate_artifact(candidate, step.payload)
                    record(step.kind, "completed", {"step_id": step.step_id})
                    continue
                if step.kind != "replay_cell":
                    skipped.append(step.step_id)
                    issues.append(
                        {
                            "type": "unknown_step",
                            "step_id": step.step_id,
                            "kind": step.kind,
                        }
                    )
                    record(step.kind, "skipped", {"step_id": step.step_id})
                    continue
                code = str(step.payload.get("code") or "")
                language = str(step.payload.get("language") or manifest.language)
                error = (
                    "step is not explicitly replay-safe"
                    if step.replay_policy != REPLAY_SAFE
                    else replay_safety_error(
                        code,
                        language=language,
                        declared_host_methods=step.payload.get("host_methods") or (),
                    )
                )
                if error:
                    skipped.append(step.step_id)
                    issues.append(
                        {
                            "type": "non_replayable",
                            "step_id": step.step_id,
                            "reason": error,
                        }
                    )
                    record(
                        "replay",
                        "skipped",
                        {"step_id": step.step_id, "reason": error},
                    )
                    continue
                response = self._execute(candidate, code, language)
                if response.get("error"):
                    issues.append(
                        {
                            "type": "replay_failed",
                            "step_id": step.step_id,
                            "error": str(response["error"]),
                        }
                    )
                    record(
                        "replay",
                        "failed",
                        {"step_id": step.step_id, "error": str(response["error"])},
                    )
                    break
                replayed.append(step.step_id)
                record("replay", "completed", {"step_id": step.step_id})

            check_cancelled()
            issues.extend(self._validate(candidate, manifest, recipe))
            if issues:
                record("validate", "partial", {"issues": issues})
                self._shutdown(candidate)
                return RecoveryResult(
                    recovery_id,
                    "partial",
                    source_generation_id,
                    candidate_id,
                    manifest.manifest_id,
                    tuple(issues),
                    tuple(replayed),
                    tuple(skipped),
                )

            record("validate", "completed")
            check_cancelled()
            self._publish(candidate)
            published = True
            record("publish", "completed")
            return RecoveryResult(
                recovery_id,
                "active",
                source_generation_id,
                candidate_id,
                manifest.manifest_id,
                (),
                tuple(replayed),
                tuple(skipped),
            )
        except RecoveryCancelled as error:
            record("restore", "cancelled", {"error": str(error)})
            if candidate is not None and not published:
                self._shutdown(candidate)
            return RecoveryResult(
                recovery_id,
                "cancelled",
                source_generation_id,
                candidate_id,
                manifest.manifest_id,
                ({"type": "recovery_cancelled", "error": str(error)},),
                tuple(replayed),
                tuple(skipped),
            )
        except Exception as error:  # noqa: BLE001 — failure is durable state
            record(
                "restore",
                "failed",
                {"error": str(error), "error_type": type(error).__name__},
            )
            if candidate is not None and not published:
                self._shutdown(candidate)
            return RecoveryResult(
                recovery_id,
                "active" if published else "failed",
                source_generation_id,
                candidate_id,
                manifest.manifest_id,
                (
                    {
                        "type": (
                            "publish_journal_failed"
                            if published
                            else "recovery_failed"
                        ),
                        "error": str(error),
                    },
                ),
                tuple(replayed),
                tuple(skipped),
            )

    def _validate(
        self,
        candidate: Candidate,
        manifest: BootstrapManifest,
        recipe: RecoveryRecipe,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if recipe.namespace_coverage == "unverified":
            issues.append(
                {
                    "type": "namespace_unverified",
                    "reason": (
                        "checkpoint contains prior Cells without a verified "
                        "namespace recovery recipe"
                    ),
                }
            )
        for language, required in recipe.required_symbols.items():
            observed = {str(item) for item in self._symbols(candidate, language)}
            missing = sorted(set(required) - observed)
            if missing:
                issues.append(
                    {"type": "missing_symbols", "language": language, "names": missing}
                )
        for artifact, expected in recipe.artifact_hashes.items():
            observed = self._artifact_digest(candidate, artifact)
            if observed != expected:
                issues.append(
                    {
                        "type": "artifact_hash_mismatch",
                        "artifact": artifact,
                        "expected": expected,
                        "observed": observed,
                    }
                )
        environment = dict(self._environment(candidate) or {})
        for key, expected in recipe.environment_requirements.items():
            if environment.get(key) != expected:
                issues.append(
                    {
                        "type": "environment_mismatch",
                        "key": key,
                        "expected": expected,
                        "observed": environment.get(key),
                    }
                )
        if environment.get("interpreter") not in {None, manifest.interpreter}:
            issues.append(
                {
                    "type": "environment_mismatch",
                    "key": "interpreter",
                    "expected": manifest.interpreter,
                    "observed": environment.get("interpreter"),
                }
            )
        observed_runtime = environment.get("runtime_version") or environment.get(
            "python_version"
        )
        if (
            manifest.runtime_version
            and manifest.runtime_version not in {"?", "unknown"}
            and observed_runtime != manifest.runtime_version
        ):
            issues.append(
                {
                    "type": "environment_mismatch",
                    "key": "runtime_version",
                    "expected": manifest.runtime_version,
                    "observed": observed_runtime,
                }
            )
        for key, expected in (
            ("sdk_version", manifest.sdk_version),
            ("provenance_version", manifest.provenance_version),
        ):
            if expected is not None and environment.get(key) != expected:
                issues.append(
                    {
                        "type": "environment_mismatch",
                        "key": key,
                        "expected": expected,
                        "observed": environment.get(key),
                    }
                )
        return issues

    @staticmethod
    def _shutdown(candidate: Candidate) -> None:
        try:
            candidate.shutdown()
        except Exception:  # noqa: BLE001 — candidate was never published
            pass


def sidecar_from_path(
    name: str,
    path: str | Path,
    *,
    order: int,
    exports: Sequence[str] = (),
    import_mode: str = "module",
) -> SidecarManifest:
    source_path = Path(path).resolve()
    return SidecarManifest(
        name=name,
        source=source_path.read_bytes(),
        order=order,
        exports=tuple(exports),
        import_mode=import_mode,
        source_path=str(source_path),
    )


__all__ = [
    "BootstrapManifest",
    "Candidate",
    "KernelRecoveryOrchestrator",
    "REPLAY_CONDITIONAL",
    "REPLAY_NEVER",
    "REPLAY_SAFE",
    "RecoveryRecipe",
    "RecoveryCancelled",
    "RecoveryResult",
    "RecoveryStep",
    "SidecarManifest",
    "replay_safety_error",
    "sidecar_from_path",
]
