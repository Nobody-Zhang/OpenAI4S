"""Web adapters that project :class:`AgentEngine` onto gateway contracts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai4s.agent.actions import (
    MULTI_CELL_NOTE,
    NO_CODE_NUDGE,
    Action,
    CodeCell,
    NativeToolBatch,
    count_code_blocks,
)
from openai4s.agent.control import execute_native_batch
from openai4s.agent.events import AgentEvent, ReplyReceived, TextDelta, TurnStarted
from openai4s.agent.models import ExecutionOutcome, ModelReply, RunState
from openai4s.agent.runtime import format_observation
from openai4s.tools import (
    MAX_TOOL_CALLS_PER_TURN,
    execute_tool_call,
    finalize_tool_batch,
    parse_fence_delimiter,
    parse_tool_calls,
    strip_fenced_blocks,
)


def _never_cancelled() -> bool:
    return False


class ProseStreamer:
    """Stream narration while hiding every top-level fenced block."""

    def __init__(self, send: Callable[[dict], None], root_frame_id: str):
        self.send = send
        self.rid = root_frame_id
        self.acc = ""
        self.line_buf = ""
        self.fence_stack: list[tuple[str, int]] = []
        self.emitted_any = False
        self.emitted = ""

    def feed(self, delta: str) -> None:
        self.acc += delta
        self.line_buf += delta
        out: list[str] = []
        while True:
            newline = self.line_buf.find("\n")
            if newline < 0:
                break
            line = self.line_buf[: newline + 1]
            self.line_buf = self.line_buf[newline + 1 :]
            delimiter = parse_fence_delimiter(line)
            if delimiter:
                fence_char, fence_length, info = delimiter
                if not self.fence_stack:
                    self.fence_stack.append((fence_char, fence_length))
                elif (
                    fence_char != self.fence_stack[-1][0]
                    or fence_length < self.fence_stack[-1][1]
                ):
                    pass
                elif info:
                    self.fence_stack.append((fence_char, fence_length))
                else:
                    self.fence_stack.pop()
            elif not self.fence_stack:
                out.append(line)
        self._emit_prose("".join(out))

    def _emit_prose(self, chunk: str) -> None:
        if not chunk:
            return
        self.send(
            {
                "type": "text_chunk",
                "frame_id": self.rid,
                "block_type": "text",
                "chunk": chunk,
            }
        )
        self.emitted += chunk
        self.emitted_any = True

    def finalize(self) -> None:
        target = strip_fenced_blocks(self.acc)
        if target.startswith(self.emitted) and len(target) > len(self.emitted):
            self._emit_prose(target[len(self.emitted) :])


@dataclass
class WebEventSink:
    """Translate typed engine events to stable WebSocket/store projections."""

    send: Callable[[dict], None]
    root_frame_id: str
    assistant_visible: list[dict]
    add_usage: Callable[[dict], None]
    current_prose: str = field(default="", init=False)
    _streamer: ProseStreamer | None = field(default=None, init=False)

    def emit(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self.current_prose = ""
            self._streamer = ProseStreamer(self.send, self.root_frame_id)
        elif isinstance(event, TextDelta):
            self._ensure_streamer().feed(event.text)
        elif isinstance(event, ReplyReceived):
            streamer = self._ensure_streamer()
            streamer.finalize()
            if event.reply.usage:
                self.add_usage(event.reply.usage)
            prose = strip_fenced_blocks(event.reply.content).strip()
            self.current_prose = prose
            if prose:
                self.assistant_visible.append(
                    {"at": int(time.time() * 1000) - 1, "text": prose}
                )
                if not streamer.emitted_any:
                    self.send(
                        {
                            "type": "text_chunk",
                            "frame_id": self.root_frame_id,
                            "block_type": "text",
                            "chunk": prose + "\n",
                        }
                    )

    def _ensure_streamer(self) -> ProseStreamer:
        if self._streamer is None:
            self._streamer = ProseStreamer(self.send, self.root_frame_id)
        return self._streamer


@dataclass
class EventCancellation:
    event: Any

    def cancelled(self) -> bool:
        return bool(self.event.is_set())


@dataclass
class WebActionExecutor:
    """Execute routed actions through the gateway's persistent session runtime."""

    dispatcher: Callable[[], Any]
    apply_pending: Callable[[], None]
    execute_cell: Callable[[CodeCell], dict]
    events: WebEventSink
    prose_nudge: str
    explore_nudge: str
    explore_mode: bool = False
    plan_mode: bool = False
    finalize_plan: Callable[[ModelReply, str], None] | None = None
    cancelled: Callable[[], bool] = _never_cancelled

    def execute(
        self, action: Action | None, reply: ModelReply, state: RunState
    ) -> ExecutionOutcome:
        del state
        if self.cancelled():
            if isinstance(action, NativeToolBatch):
                return self._refuse_native(
                    action, "run was cancelled before execution", "cancelled"
                )
            return ExecutionOutcome(stop_reason="cancelled")
        if self.plan_mode:
            return self._capture_plan(action, reply)
        if isinstance(action, NativeToolBatch):
            outcome = execute_native_batch(
                action, self._invoke_native, cancelled=self.cancelled
            )
            if self.cancelled():
                return outcome
            return self._apply_trailing_pending(outcome)
        if isinstance(action, CodeCell):
            self.apply_pending()
            result = self.execute_cell(action)
            observation = format_observation(result)
            if count_code_blocks(reply.content) > 1:
                observation += MULTI_CELL_NOTE
            completion = getattr(self.dispatcher(), "last_output", None)
            return self._user_observation(observation, completion=completion)
        return self._legacy_or_nudge(reply)

    def _capture_plan(
        self, action: Action | None, reply: ModelReply
    ) -> ExecutionOutcome:
        if self.finalize_plan is not None:
            self.finalize_plan(reply, self.events.current_prose)
        if isinstance(action, NativeToolBatch):
            return self._refuse_native(
                action, "tools are disabled in plan mode", "plan"
            )
        return ExecutionOutcome(stop_reason="plan")

    @staticmethod
    def _refuse_native(
        action: NativeToolBatch, reason: str, stop_reason: str
    ) -> ExecutionOutcome:
        refused = execute_native_batch(
            action,
            lambda call: (f"[Tool error] {call.name}: {reason}", False),
            limit=len(action.calls),
        )
        return ExecutionOutcome(
            refused.history_messages,
            observation=refused.observation,
            stop_reason=stop_reason,
        )

    def _invoke_native(self, call) -> tuple[str, bool]:
        self.apply_pending()
        return execute_tool_call(
            self.dispatcher(),
            {"name": call.name, "arguments": call.arguments},
        )

    def _apply_trailing_pending(
        self, outcome: ExecutionOutcome
    ) -> ExecutionOutcome:
        try:
            self.apply_pending()
            return outcome
        except Exception as exc:  # noqa: BLE001 — keep native history replayable
            notice = f"[Tool error] pending environment switch failed: {exc}"
            history = [dict(message) for message in outcome.history_messages]
            if history:
                target = next(
                    (
                        index
                        for index in range(len(history) - 1, -1, -1)
                        if not history[index].get("is_error")
                    ),
                    len(history) - 1,
                )
                history[target]["content"] += "\n" + notice
                history[target]["is_error"] = True
            observation = str(outcome.observation or "") + "\n" + notice
            return ExecutionOutcome(tuple(history), observation=observation)

    def _legacy_or_nudge(self, reply: ModelReply) -> ExecutionOutcome:
        calls, errors = parse_tool_calls(reply.content)
        if calls or errors:
            parts: list[str] = []
            for call in calls[:MAX_TOOL_CALLS_PER_TURN]:
                if self.cancelled():
                    errors.append("remaining legacy tool calls skipped: cancelled")
                    break
                try:
                    self.apply_pending()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"pending environment switch failed: {exc}")
                    break
                text, _ok = execute_tool_call(self.dispatcher(), call)
                parts.append(text)
            if not self.cancelled():
                try:
                    self.apply_pending()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"pending environment switch failed: {exc}")
            observation = finalize_tool_batch(parts, len(calls), errors)
        elif self.events.current_prose:
            observation = self.explore_nudge if self.explore_mode else self.prose_nudge
        else:
            observation = NO_CODE_NUDGE
        return self._user_observation(observation)

    @staticmethod
    def _user_observation(
        observation: str, *, completion: Any = None
    ) -> ExecutionOutcome:
        return ExecutionOutcome(
            ({"role": "user", "content": observation},),
            observation=observation,
            completion=completion,
        )


__all__ = [
    "EventCancellation",
    "ProseStreamer",
    "WebActionExecutor",
    "WebEventSink",
]
