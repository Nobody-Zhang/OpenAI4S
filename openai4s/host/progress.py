"""Session todo and approved-plan progress behavior for host RPC calls."""

from __future__ import annotations

from typing import Callable, Protocol

PLAN_STEP_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "failed", "skipped"}
)
PlanSink = Callable[[dict], None]


class ProgressStore(Protocol):
    def get_plan(self, plan_id: str) -> dict | None: ...

    def get_plan_by_frame(self, frame_id: str) -> dict | None: ...

    def set_plan_step_status(
        self,
        plan_id: str,
        step_id: str,
        status: str,
        note: str | None = None,
    ) -> dict | None: ...


class ProgressService:
    """Own one dispatcher's transient todos and persisted plan-step ticks."""

    def __init__(
        self,
        store: ProgressStore,
        *,
        get_frame_id: Callable[[], str | None],
        get_plan_sink: Callable[[], PlanSink | None],
    ) -> None:
        self.store = store
        self.get_frame_id = get_frame_id
        self.get_plan_sink = get_plan_sink
        self._todos: list[dict] = []

    def todo_write(self, spec: dict) -> dict:
        todos = spec.get("todos") or []
        clean = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            clean.append(
                {
                    "id": todo.get("id") or f"t{len(clean) + 1}",
                    "content": todo.get("content", ""),
                    "status": todo.get("status", "pending"),
                    "priority": todo.get("priority", "medium"),
                }
            )
        self._todos = clean
        return {"ok": True, "count": len(clean), "todos": clean}

    def todo_read(self) -> dict:
        return {"todos": self._todos}

    def plan_update(self, spec: dict) -> dict:
        step_id = spec.get("step_id") or spec.get("id")
        status = spec.get("status") or "in_progress"
        if status not in PLAN_STEP_STATUSES:
            status = "in_progress"
        note = spec.get("note")
        plan_id = spec.get("plan_id")
        if plan_id:
            plan = self.store.get_plan(plan_id)
        else:
            frame_id = self.get_frame_id()
            plan = self.store.get_plan_by_frame(frame_id) if frame_id else None
        if not plan:
            return {"error": "no active plan for this session"}
        if not step_id:
            return {"error": "plan_update requires step_id"}

        self.store.set_plan_step_status(plan["plan_id"], step_id, status, note)
        sink = self.get_plan_sink()
        if sink is not None:
            try:
                sink(
                    {
                        "plan_id": plan["plan_id"],
                        "step_id": step_id,
                        "status": status,
                        "note": note,
                    }
                )
            except Exception:  # noqa: BLE001 - progress telemetry is best effort
                pass
        return {
            "ok": True,
            "plan_id": plan["plan_id"],
            "step_id": step_id,
            "status": status,
        }

    def plan_read(self) -> dict:
        frame_id = self.get_frame_id()
        plan = self.store.get_plan_by_frame(frame_id) if frame_id else None
        return plan or {"plan": None}


__all__ = ["PLAN_STEP_STATUSES", "ProgressService"]
