"""Read-only execution and lineage projections for the Web UI."""

from __future__ import annotations

from typing import Callable, Protocol

from openai4s.agent.actions import is_completion_only_cell


class ExecutionViewStore(Protocol):
    def list_cells(self, root_frame_id: str) -> list[dict]: ...

    def get_artifact(self, artifact_id: str) -> dict | None: ...

    def version_meta(self, version_id: str) -> dict | None: ...

    def lineage_inputs(self, version_id: str) -> list[dict]: ...

    def cell_detail(self, producing_cell_id: str) -> dict | None: ...


class ExecutionViewService:
    """Project persisted execution records into Notebook/Provenance DTOs."""

    def __init__(
        self,
        *,
        store: ExecutionViewStore,
        format_timestamp: Callable[[int | float | None], str | None],
    ) -> None:
        self.store = store
        self.format_timestamp = format_timestamp

    def execution_log(self, root_frame_id: str) -> dict:
        kernels: list[str] = []
        entries = []
        for cell in self.store.list_cells(root_frame_id):
            language = cell.get("language") or "python"
            if is_completion_only_cell(cell.get("code") or "", language):
                continue
            kernel_id = cell.get("kernel_id") or "python"
            if kernel_id not in kernels:
                kernels.append(kernel_id)
            entries.append(
                {
                    "cell_index": cell.get("cell_index"),
                    "kernel_id": kernel_id,
                    "language": language,
                    "source": cell.get("code") or "",
                    "stdout": cell.get("stdout") or "",
                    "stderr": cell.get("stderr") or "",
                    "error": cell.get("error") or "",
                    "status": cell.get("status") or "ok",
                    "figures": cell.get("figures") or [],
                    "files_written": cell.get("files_written") or [],
                    "files_read": cell.get("files_read") or [],
                    "cpu_seconds": cell.get("cpu_s"),
                    "peak_rss_kb": cell.get("peak_rss_kb"),
                }
            )
        return {"kernels": kernels, "entries": entries}

    def artifact_lineage(self, artifact_id: str) -> dict:
        artifact = self.store.get_artifact(artifact_id)
        if not artifact:
            return {
                "artifact_id": artifact_id,
                "filename": None,
                "interactions": [],
                "dependency_mappings": {"inputs": []},
            }

        interactions = []
        version_id = artifact.get("latest_version_id")
        cell = None
        version = None
        edge_inputs: list[str] = []
        if version_id:
            version = self.store.version_meta(version_id)
            for item in self.store.lineage_inputs(version_id):
                label = (
                    item.get("filename")
                    or item.get("path")
                    or item.get("version_id")
                )
                if label:
                    edge_inputs.append(str(label))
            producing_cell_id = (version or {}).get("producing_cell_id")
            if producing_cell_id:
                cell = self.store.cell_detail(producing_cell_id)

        files_written: list[str] = []
        legacy_reads: list[str] = []
        if cell:
            files_written = cell.get("files_written") or []
            legacy_reads = cell.get("files_read") or []

        known_reads: list[str] = []
        seen_reads: set[str] = set()
        for filename in [*legacy_reads, *edge_inputs]:
            if filename and filename not in seen_reads:
                seen_reads.add(filename)
                known_reads.append(filename)

        outputs = set(files_written)
        outputs.add(artifact["filename"])
        inputs = [filename for filename in known_reads if filename not in outputs]
        if cell:
            interactions.append(
                {
                    "kind": "cell",
                    "cell_index": cell.get("cell_index"),
                    "kernel_id": cell.get("kernel_id") or "python",
                    "language": cell.get("language") or "python",
                    "exit_status": cell.get("status") or "ok",
                    "source": cell.get("code") or "",
                    "files_written": files_written,
                    "files_read": known_reads,
                }
            )
        interactions.append(
            {
                "kind": "save",
                "at": self.format_timestamp(
                    (version or {}).get("created_at")
                    or artifact.get("created_at")
                ),
            }
        )
        return {
            "artifact_id": artifact_id,
            "filename": artifact.get("filename"),
            "interactions": interactions,
            "dependency_mappings": {"inputs": inputs},
        }


__all__ = ["ExecutionViewService"]
