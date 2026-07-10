"""Local runtime adapters for the provider-neutral :mod:`agent.engine`.

The engine owns the turn state machine.  This module connects it to the
blocking LLM client, context compaction, persistent kernels, and the existing
dispatcher-backed control tools without importing those concrete services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from openai4s.tools import (
    MAX_TOOL_CALLS_PER_TURN,
    execute_tool_call,
    parse_tool_calls,
    run_tool_calls,
)

from .actions import (
    MULTI_CELL_NOTE,
    NO_CODE_NUDGE,
    Action,
    CodeCell,
    NativeToolBatch,
    count_code_blocks,
)
from .compaction import compact, safe_keep_recent, should_compact
from .control import execute_native_batch
from .events import AgentEvent, OutcomeProduced, ReplyReceived
from .models import ExecutionOutcome, ModelReply, RunState

LogFn = Callable[..., None]


def _null_log(*args: object) -> None:
    del args


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    content: str


@dataclass
class CompletionSignal:
    read: Callable[[], Any]

    def completion(self) -> Any:
        return self.read()


@dataclass
class ChatModel:
    """Adapt the blocking ``chat`` function to ``ModelPort``."""

    cfg: Any
    chat_fn: Callable[..., Mapping[str, Any]]
    tools: Sequence[Any] = ()
    stream: bool = False

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_delta: Callable[[str], None],
    ) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {"tools": tuple(self.tools)}
        if self.stream:
            kwargs["on_delta"] = on_delta
        return self.chat_fn([dict(message) for message in messages], self.cfg, **kwargs)


@dataclass
class CompactionPolicy:
    """Apply token-triggered compaction while preserving tool batches."""

    cfg: Any
    log: LogFn = _null_log

    def prepare(self, state: RunState) -> Sequence[Mapping[str, Any]]:
        messages = state.messages
        if not should_compact(messages, self.cfg):
            return messages
        prepared = compact(
            messages,
            self.cfg,
            keep_recent=safe_keep_recent(messages),
            archive_dir=self.cfg.compaction_dir,
        )
        self.log(f"[compacted] messages -> {len(prepared)}")
        return prepared


@dataclass
class LocalActionExecutor:
    """Execute one selected action against a run-scoped local runtime."""

    kernel: Any
    dispatcher: Any
    pre_exec_gate: Callable[[str, list[dict]], str | None]
    execute_r: Callable[[str], dict]
    log: LogFn = _null_log

    def execute(
        self, action: Action | None, reply: ModelReply, state: RunState
    ) -> ExecutionOutcome:
        if isinstance(action, NativeToolBatch):
            return self._execute_native(action)
        if isinstance(action, CodeCell):
            return self._execute_code(action, reply, state)
        return self._execute_legacy_or_nudge(reply)

    def _execute_native(self, batch: NativeToolBatch) -> ExecutionOutcome:
        def invoke(call):
            return execute_tool_call(
                self.dispatcher,
                {"name": call.name, "arguments": call.arguments},
            )

        return execute_native_batch(batch, invoke)

    def _execute_code(
        self, action: CodeCell, reply: ModelReply, state: RunState
    ) -> ExecutionOutcome:
        refusal = self.pre_exec_gate(action.code, state.messages)
        if refusal is not None:
            self.log(f"[safety] cell not executed: {refusal}")
            return self._user_observation(refusal)
        if action.language == "r":
            result = self.execute_r(action.code)
        else:
            result = self.kernel.execute(action.code, origin="agent")
        observation = format_observation(result)
        if count_code_blocks(reply.content) > 1:
            observation += MULTI_CELL_NOTE
        completion = getattr(self.dispatcher, "last_output", None)
        return self._user_observation(observation, completion=completion)

    def _execute_legacy_or_nudge(self, reply: ModelReply) -> ExecutionOutcome:
        calls, errors = parse_tool_calls(reply.content)
        if calls or errors:
            observation = run_tool_calls(self.dispatcher, calls, errors)
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


@dataclass
class TranscriptEventSink:
    """Project typed engine events onto the stable CLI transcript."""

    transcript: list[TranscriptTurn]
    log: LogFn = _null_log

    def emit(self, event: AgentEvent) -> None:
        if isinstance(event, ReplyReceived):
            self.transcript.append(TranscriptTurn("assistant", event.reply.content))
            self.log(
                f"\n--- turn {event.turn} (assistant) ---\n{event.reply.content}"
            )
        elif (
            isinstance(event, OutcomeProduced)
            and event.outcome.observation is not None
        ):
            content = str(event.outcome.observation)
            self.transcript.append(TranscriptTurn("observation", content))
            self.log(f"--- turn {event.turn} (observation) ---\n{content}")


def format_observation(result: dict) -> str:
    """Format one kernel result as the stable observation protocol."""
    parts = ["[Observation]"]
    out = result.get("stdout") or ""
    err = result.get("stderr") or ""
    error = result.get("error")
    if out:
        parts.append(f"stdout:\n{out.rstrip()}")
    if err:
        parts.append(f"stderr:\n{err.rstrip()}")
    if error:
        trace = result.get("trace") or {}
        line = trace.get("error_lineno")
        location = f" (cell line {line})" if line else ""
        parts.append(f"ERROR{location}:\n{error.rstrip()}")
    if not out and not err and not error:
        parts.append("(no output)")
    usage = result.get("usage") or {}
    if usage:
        parts.append(
            f"[usage wall={usage.get('wall_s')}s "
            f"cpu={usage.get('cpu_s')}s rss={usage.get('peak_rss_kb')}kb]"
        )
    return "\n".join(parts)


__all__ = [
    "ChatModel",
    "CompactionPolicy",
    "CompletionSignal",
    "LocalActionExecutor",
    "TranscriptEventSink",
    "TranscriptTurn",
    "format_observation",
]
