"""User-visible projections for control actions and structured completion."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable
from urllib.parse import quote

from openai4s.agent.actions import (
    Action,
    CodeCell,
    NativeToolBatch,
    is_completion_only_cell,
)

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SUMMARY_KEYS = (
    "summary",
    "answer",
    "conclusion",
    "message",
    "result",
    "摘要",
    "结论",
    "回答",
)


def response_language(text: Any) -> str:
    """Choose the deterministic fallback language from the user's text."""
    return "zh" if _CJK.search(str(text or "")) else "en"


def action_narration(action: Action | None, language: str = "en") -> str:
    """Describe an action without inventing reasoning or scientific results."""
    zh = language == "zh"
    if isinstance(action, CodeCell):
        if is_completion_only_cell(action):
            return ""
        if action.language == "r":
            return (
                "我正在用 R 完成这一阶段的分析，关键输出会保留在 Notebook 中。"
                if zh
                else "I am running this analysis stage in R; its key outputs will remain in the Notebook."
            )
        return (
            "我正在运行这一阶段的分析，关键输出会保留在 Notebook 和最终结果中。"
            if zh
            else "I am running this analysis stage; key outputs will remain in the Notebook and final result."
        )
    if not isinstance(action, NativeToolBatch) or not action.calls:
        return ""

    names = {call.name for call in action.calls}
    if len(action.calls) > 1:
        return (
            f"我正在执行 {len(action.calls)} 个相关步骤，并会根据返回结果继续分析。"
            if zh
            else f"I am running {len(action.calls)} related steps and will continue from their returned results."
        )
    name = action.calls[0].name
    if name == "web_search":
        return (
            "我先检索相关来源，随后会说明检索结果如何影响下一步分析。"
            if zh
            else "I am searching relevant sources first, then I will explain how the results affect the next analysis step."
        )
    if name == "web_fetch":
        return (
            "我正在读取并核对这个来源中的关键证据。"
            if zh
            else "I am reading this source and checking its key evidence."
        )
    if names & {"write_file", "edit_file"}:
        return (
            "我正在保存阶段性结果，写出的文件会加入 Artifacts。"
            if zh
            else "I am saving the current results; written files will be added to Artifacts."
        )
    if names & {"read_text_file", "list_dir", "glob_files", "content_search"}:
        return (
            "我先检查相关文件和数据，再根据实际内容继续分析。"
            if zh
            else "I am inspecting the relevant files and data before continuing from their actual contents."
        )
    if names & {"env_list", "env_use", "env_create"}:
        return (
            "我正在准备适合这一步分析的运行环境。"
            if zh
            else "I am preparing the appropriate runtime environment for this analysis step."
        )
    return (
        "我正在执行下一步，并会把关键结果反馈在对话中。"
        if zh
        else "I am running the next step and will report its key result in the conversation."
    )


def completion_message(
    completion: Any,
    artifacts: Iterable[dict] = (),
    *,
    previous_text: str = "",
    language: str = "en",
    require_fallback: bool = True,
) -> str:
    """Render a submitted result into a durable, human-facing final message."""
    spec = completion if isinstance(completion, dict) else {}
    output = spec.get("output", completion)
    bullets = spec.get("completion_bullets") or []
    zh = language == "zh"
    previous = _normalized(previous_text)
    parts: list[str] = []

    summary = _summary_text(output)
    if summary and not _summary_already_visible(output, summary, previous):
        parts.append(summary)

    fresh_bullets = [
        str(item).strip()
        for item in bullets
        if str(item).strip() and _normalized(str(item)) not in previous
    ]
    if fresh_bullets:
        heading = "完成内容：" if zh else "Completed work:"
        parts.append(heading + "\n" + "\n".join(f"- {item}" for item in fresh_bullets))

    artifact_lines = _artifact_lines(artifacts)
    if artifact_lines:
        heading = "产物：" if zh else "Artifacts:"
        parts.append(heading + "\n" + "\n".join(artifact_lines))

    if not parts and previous:
        return ""
    if not parts and require_fallback:
        return "任务已完成。" if zh else "The task is complete."
    return "\n\n".join(parts)


def _summary_text(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if not isinstance(output, dict):
        if output is None:
            return ""
        return str(output).strip()
    for key in _SUMMARY_KEYS:
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return f"{key}: {value}"
    visible = {
        str(key): value
        for key, value in output.items()
        if key not in {"artifact", "artifacts", "report_file", "file", "files"}
    }
    if not visible or set(visible) <= {"ok", "status"}:
        return ""
    rendered = json.dumps(visible, ensure_ascii=False, indent=2, default=str)
    return f"```json\n{rendered[:4000]}\n```"


def _summary_already_visible(output: Any, summary: str, previous: str) -> bool:
    if _normalized(summary) in previous:
        return True
    if not isinstance(output, dict):
        return False
    for key in _SUMMARY_KEYS:
        value = output.get(key)
        normalized = _normalized(str(value)) if value is not None else ""
        if normalized and normalized in previous:
            return True
    return False


def _artifact_lines(artifacts: Iterable[dict]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        name = str(artifact.get("filename") or "artifact")
        ident = str(artifact.get("artifact_id") or artifact.get("id") or "")
        key = (ident, name)
        if key in seen:
            continue
        seen.add(key)
        target = quote(ident or name, safe="")
        label = name.replace("[", "\\[").replace("]", "\\]")
        lines.append(f"- [{label}](/api/artifacts/{target})")
    return lines


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


__all__ = ["action_narration", "completion_message", "response_language"]
