"""Hybrid orchestration engine and backward-compatible local Agent facade."""

from openai4s.agent.engine import AgentEngine
from openai4s.agent.finalize import (
    CompletionRecord,
    finalize_response_tool_spec,
    with_finalize_response,
)
from openai4s.agent.loop import Agent, run_task
from openai4s.agent.models import EngineResult, ExecutionOutcome, ModelReply, RunState

__all__ = [
    "Agent",
    "AgentEngine",
    "CompletionRecord",
    "EngineResult",
    "ExecutionOutcome",
    "ModelReply",
    "RunState",
    "finalize_response_tool_spec",
    "run_task",
    "with_finalize_response",
]
