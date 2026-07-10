"""Shared action-parsing core for the dual outer loop.

Both turn loops — ``Agent.run`` (openai4s/agent/loop.py) and
``SessionRunner._loop`` (openai4s/server/gateway.py) — parse a model reply into
at most ONE executable action per step. This module is the single choke point
for that decision (CoreCoder-style: one small core, two thin loop bodies):

- the fence-info → language whitelist (``python``/``py``/bare → python kernel,
  ``r``/``R`` → R kernel) lives only here;
- the code-wins-over-tools precedence stays with the callers, but both share
  the same extractor so a quoted ```` ```tool ```` inside a cell can never
  hijack a turn in one loop and not the other;
- the one-cell-per-step counter and the no-action nudge text live here so the
  two loops cannot drift.

The host executes exactly two kinds of instructions — python cells on the
persistent Jupyter-style kernel and R cells on the persistent R kernel. Any
other work (shell, file ops) happens *inside* those kernels or through the
declarative ```tool surface.

Pure stdlib; imports only openai4s.tools (itself pure stdlib, zero side
effects on import).
"""
from __future__ import annotations

from dataclasses import dataclass

from openai4s.tools import scan_fenced_blocks

# Fence info strings (already lowercased by parse_fence_delimiter) that mark an
# executable cell. A bare ``` fence still means python — R must be explicit.
PYTHON_INFOS = ("", "python", "py")
R_INFOS = ("r",)


@dataclass(frozen=True)
class CodeCell:
    """One executable cell extracted from a model reply."""

    language: str  # "python" | "r"
    code: str


def extract_action(text: str) -> CodeCell | None:
    """Return the FIRST complete top-level executable cell in a model reply.

    Document order decides between languages: whichever labelled (or bare)
    executable fence closes first wins, and exactly one cell runs per step.
    The shared fence scanner preserves labelled fenced examples nested inside
    the cell (notably a literal ```tool block in a triple-quoted README). An
    incomplete outer fence is never executable.
    """
    for block in scan_fenced_blocks(text):
        if not (block.closed and block.fence_char == "`"):
            continue
        if block.info in PYTHON_INFOS:
            return CodeCell("python", block.body)
        if block.info in R_INFOS:
            return CodeCell("r", block.body)
    return None


def count_code_blocks(text: str) -> int:
    """Closed top-level executable cells (both languages) in a reply — feeds
    the one-cell-per-step note when a model batches several cells."""
    n = 0
    for block in scan_fenced_blocks(text):
        if (
            block.closed
            and block.fence_char == "`"
            and (block.info in PYTHON_INFOS or block.info in R_INFOS)
        ):
            n += 1
    return n


# Fed back when a working turn contains neither a cell nor a tool call.
NO_CODE_NUDGE = (
    "[system] No code cell found. Reply with a ```python cell (or ```r for R) "
    "to act, and call host.submit_output(...) from a python cell when the "
    "task is done."
)

# Appended to the observation when a reply batched several cells (only the
# first one ran).
MULTI_CELL_NOTE = (
    "\n[system] NOTE: only the FIRST code cell in your reply was executed — "
    "exactly ONE cell runs per step. The later cells did NOT run, and any "
    "results you described for them are not real. Do not assume they "
    "succeeded: continue with the NEXT single cell based on the real "
    "observation above."
)
