"""Compile immutable Cell facts into a conservative recovery recipe.

Recovery is not Notebook replay.  This module selects only successful Cells
whose namespace effects can be reconstructed without repeating an external
write, an unknown Host action, or an uncertain namespace mutation.  Every
state-affecting Cell that cannot pass those checks remains in the recipe as a
``never`` replay step, so validation reports ``RecoveryPartial`` instead of
silently claiming that the old namespace survived.

The compiler is deliberately stdlib-only and side-effect free.  It consumes
the immutable execution-log DTOs plus bootstrap manifests already captured by
the session domain; worker construction and replay stay in ``kernel.recovery``.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from openai4s.execution.dependencies import normalize_string_list
from openai4s.kernel.recovery import BootstrapManifest, replay_safety_error

_PYTHON_RUNTIME_SYMBOLS = frozenset(dir(builtins)) | frozenset(
    {
        "__builtins__",
        "__doc__",
        "__name__",
        "__package__",
        "host",
    }
)

# The dependency-free R lexer intentionally reports function identifiers as
# reads.  These names are supplied by a pristine base R worker and therefore do
# not require a producing Notebook Cell.  Package functions are deliberately
# absent: their library/import Cell must be represented, otherwise recovery is
# partial rather than guessing which mutable package set was attached.
_R_RUNTIME_SYMBOLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "as.character",
        "as.data.frame",
        "as.double",
        "as.integer",
        "as.list",
        "as.logical",
        "as.matrix",
        "as.numeric",
        "attributes",
        "c",
        "cat",
        "ceiling",
        "class",
        "colMeans",
        "colSums",
        "data.frame",
        "dim",
        "exp",
        "floor",
        "head",
        "length",
        "list",
        "log",
        "matrix",
        "max",
        "mean",
        "min",
        "names",
        "ncol",
        "nrow",
        "paste",
        "paste0",
        "print",
        "range",
        "rep",
        "rev",
        "round",
        "rowMeans",
        "rowSums",
        "seq",
        "seq_along",
        "seq_len",
        "sort",
        "sqrt",
        "stop",
        "sum",
        "summary",
        "tail",
        "unlist",
        "warning",
    }
)

_NONDETERMINISTIC_MODULES = frozenset({"datetime", "random", "secrets", "time", "uuid"})
_NONDETERMINISTIC_CALLS = frozenset(
    {
        "now",
        "perf_counter",
        "process_time",
        "random",
        "randint",
        "randrange",
        "time",
        "today",
        "token_bytes",
        "token_hex",
        "token_urlsafe",
        "uniform",
        "urandom",
        "uuid1",
        "uuid4",
    }
)


def build_recovery_recipe(
    cells: Sequence[Mapping[str, Any]],
    *,
    generation_refs: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Return a version-1 recipe derived only from durable execution facts.

    ``conditional`` is an input policy, not permission to replay.  A Cell is
    promoted to a ``safe`` recovery step only after source-hash, success,
    dependency, Host/effect, and determinism checks all pass.  Conversely, a
    failed/manual Cell with no possible namespace effect is omitted: workspace
    and Artifact hydration restore its durable outputs without repeating the
    external action.
    """

    manifests = _bootstrap_manifests(generation_refs)
    producers: dict[str, dict[str, str | None]] = {"python": {}, "r": {}}
    live_symbols: dict[str, set[str]] = {"python": set(), "r": set()}
    steps: list[dict[str, Any]] = []
    state_cells = 0
    manual_cells = 0

    for index, raw_cell in enumerate(cells):
        cell = dict(raw_cell)
        language = str(cell.get("language") or "python").strip().lower()
        reads = normalize_string_list(cell.get("variable_reads"))
        writes = normalize_string_list(cell.get("variable_writes"))
        deletes = normalize_string_list(cell.get("variable_deletes"))
        uncertain = bool(cell.get("mutation_uncertain"))
        affects_namespace = bool(writes or deletes or uncertain)
        if not affects_namespace:
            continue

        state_cells += 1
        code = str(cell.get("code") or "")
        actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        recorded_hash = str(cell.get("code_hash") or "")
        cell_id = str(
            cell.get("producing_cell_id")
            or f"cell-{cell.get('state_revision') or index + 1}"
        )
        required = tuple(sorted(set(reads) | set(deletes)))
        runtime_symbols = _runtime_symbols(language)
        language_producers = producers.setdefault(language, {})
        dependency_cells = sorted(
            {
                producer
                for name in required
                if (producer := language_producers.get(name))
            }
        )
        unresolved = sorted(
            name
            for name in required
            if name not in runtime_symbols and not language_producers.get(name)
        )
        host_methods, host_shape_error = _python_host_usage(code, language)
        reasons = _manual_reasons(
            cell,
            language=language,
            code=code,
            actual_hash=actual_hash,
            recorded_hash=recorded_hash,
            unresolved=unresolved,
            host_methods=host_methods,
            host_shape_error=host_shape_error,
            has_manifest=language in manifests,
        )

        payload: dict[str, Any] = {
            "cell_id": cell_id,
            "state_revision": cell.get("state_revision"),
            "language": language,
            "code": code,
            "code_hash": actual_hash,
            "required_symbols": list(required),
            "dependency_cells": dependency_cells,
            "produced_symbols": list(writes),
            "deleted_symbols": list(deletes),
            "host_methods": list(host_methods),
            "file_dependencies": list(normalize_string_list(cell.get("files_read"))),
        }
        step = {
            "kind": "replay_cell",
            "step_id": f"rs-cell-{index + 1}-{actual_hash[:12]}",
            "payload": payload,
            "replay_policy": "safe" if not reasons else "never",
        }
        if reasons:
            payload["manual_reasons"] = reasons
            manual_cells += 1
        steps.append(step)

        # The producer map represents what an otherwise complete replay can
        # supply to later Cells.  A manual/failed writer poisons that symbol;
        # dependent Cells remain manual even if their own source is pure.
        successful = str(cell.get("status") or "").lower() == "ok"
        for name in deletes:
            language_producers.pop(name, None)
            if successful:
                live_symbols.setdefault(language, set()).discard(name)
        for name in writes:
            language_producers[name] = cell_id if not reasons else None
            if successful:
                live_symbols.setdefault(language, set()).add(name)

    required_symbols = {
        language: sorted(names) for language, names in live_symbols.items() if names
    }
    if state_cells == 0:
        coverage = "empty"
    elif manual_cells:
        coverage = "unverified"
    else:
        coverage = "verified"

    return {
        "version": 1,
        "steps": steps,
        "required_symbols": required_symbols,
        "artifact_hashes": {
            str(name): str(digest) for name, digest in artifact_hashes.items()
        },
        "environment_requirements": _environment_requirements(manifests),
        "namespace_coverage": coverage,
        "summary": {
            "state_cells": state_cells,
            "safe_replay_cells": state_cells - manual_cells,
            "manual_cells": manual_cells,
        },
    }


def _manual_reasons(
    cell: Mapping[str, Any],
    *,
    language: str,
    code: str,
    actual_hash: str,
    recorded_hash: str,
    unresolved: Sequence[str],
    host_methods: Sequence[str],
    host_shape_error: str | None,
    has_manifest: bool,
) -> list[str]:
    reasons: list[str] = []
    if language not in {"python", "r"}:
        reasons.append(f"unsupported Cell language: {language or '<empty>'}")
    if not has_manifest:
        reasons.append(f"checkpoint has no {language} bootstrap manifest")
    if str(cell.get("status") or "").lower() != "ok":
        reasons.append("Cell did not complete successfully")
    if str(cell.get("replay_policy") or "conditional").lower() == "never":
        reasons.append("Cell replay policy is never")
    if not recorded_hash or recorded_hash != actual_hash:
        reasons.append("recorded source hash does not match Cell source")
    if bool(cell.get("mutation_uncertain")):
        reasons.append("Cell may mutate unknown namespace state")
    if normalize_string_list(cell.get("files_written")):
        reasons.append("Cell wrote external/workspace files")
    if unresolved:
        reasons.append("unresolved input symbols: " + ", ".join(unresolved))
    if host_shape_error:
        reasons.append(host_shape_error)
    safety = replay_safety_error(
        code,
        language=language,
        declared_host_methods=host_methods,
    )
    if safety:
        reasons.append(safety)
    determinism = _determinism_error(code, language)
    if determinism:
        reasons.append(determinism)
    # Stable ordering and deduplication make checkpoint hashes deterministic.
    return list(dict.fromkeys(reasons))


def _bootstrap_manifests(
    generation_refs: Mapping[str, Any],
) -> dict[str, BootstrapManifest]:
    manifests: dict[str, BootstrapManifest] = {}
    for language, ref in generation_refs.items():
        raw = (
            ref.get("bootstrap_manifest") or ref.get("bootstrap")
            if isinstance(ref, Mapping)
            else None
        )
        if not isinstance(raw, Mapping):
            continue
        try:
            manifest = BootstrapManifest.from_record(raw)
        except (TypeError, ValueError):
            continue
        if manifest.language == str(language):
            manifests[manifest.language] = manifest
    return manifests


def _environment_requirements(
    manifests: Mapping[str, BootstrapManifest],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for language, manifest in manifests.items():
        requirements = {
            str(key): value
            for key, value in manifest.environment.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }
        if manifest.runtime_version and manifest.runtime_version not in {
            "?",
            "unknown",
        }:
            requirements["runtime_version"] = manifest.runtime_version
        if manifest.sdk_version:
            requirements["sdk_version"] = manifest.sdk_version
        if manifest.provenance_version:
            requirements["provenance_version"] = manifest.provenance_version
        if manifest.host_capability_version:
            requirements["host_capability_version"] = manifest.host_capability_version
        if manifest.environment_hash:
            requirements["environment_hash"] = manifest.environment_hash
        result[language] = requirements
    return result


def _runtime_symbols(language: str) -> frozenset[str]:
    if language == "python":
        return _PYTHON_RUNTIME_SYMBOLS
    if language == "r":
        return _R_RUNTIME_SYMBOLS
    return frozenset()


def _python_host_usage(code: str, language: str) -> tuple[tuple[str, ...], str | None]:
    if language != "python":
        return (), None
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return (), None
    methods: set[str] = set()
    permitted_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "host"
        ):
            methods.add(function.attr)
            permitted_nodes.add(id(function.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "host"
            and id(node) not in permitted_nodes
        ):
            return tuple(sorted(methods)), (
                "Host object is aliased or used outside a direct host.method(...) call"
            )
    return tuple(sorted(methods)), None


def _determinism_error(code: str, language: str) -> str | None:
    if language == "r":
        if re.search(r"\b(?:Sys\.time|Sys\.Date|runif|rnorm|sample)\s*\(", code):
            return "Cell depends on uncaptured time or random state"
        return None
    if language != "python":
        return None
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    # Seed the canonical nondeterministic module names (and the ubiquitous
    # numpy alias) so a call like ``random.random()`` / ``np.random.rand()``
    # whose ``import`` lives in an EARLIER replay cell is still flagged.  The
    # per-cell scan only ever saw same-cell imports, so the ordinary "import
    # once, use many" pattern silently defeated determinism detection and let a
    # nondeterministic cell replay into a divergent namespace.
    module_aliases: dict[str, str] = {name: name for name in _NONDETERMINISTIC_MODULES}
    module_aliases["np"] = "numpy"
    module_aliases["numpy"] = "numpy"
    call_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _NONDETERMINISTIC_MODULES or alias.name == "numpy.random":
                    module_aliases[alias.asname or root] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if (
                module.split(".", 1)[0] in _NONDETERMINISTIC_MODULES
                or module == "numpy.random"
            ):
                for alias in node.names:
                    if alias.name in _NONDETERMINISTIC_CALLS or module in {
                        "random",
                        "secrets",
                        "uuid",
                        "numpy.random",
                    }:
                        call_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in call_aliases:
            return "Cell depends on uncaptured time or random state"
        if isinstance(node.func, ast.Attribute):
            chain: list[str] = []
            current: ast.AST = node.func
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chain.append(current.id)
                chain.reverse()
                root = chain[0]
                if root in module_aliases and (
                    chain[-1] in _NONDETERMINISTIC_CALLS or "random" in chain[1:]
                ):
                    return "Cell depends on uncaptured time or random state"
    return None


__all__ = ["build_recovery_recipe"]
