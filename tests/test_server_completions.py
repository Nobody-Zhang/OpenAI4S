"""Public progress and completion projections never depend on hidden reasoning."""

from openai4s.agent.actions import CodeCell, NativeToolBatch, NativeToolCall
from openai4s.server.completions import (
    action_narration,
    completion_message,
    response_language,
)


def _call(name: str) -> NativeToolCall:
    return NativeToolCall(
        id="call-1",
        wire_id="call-1",
        name=name,
        ordinal=0,
        raw_arguments='{"secret":"must-not-leak"}',
        arguments={"secret": "must-not-leak"},
    )


def test_action_narration_is_safe_localized_and_hides_raw_arguments():
    text = action_narration(NativeToolBatch((_call("web_search"),)), "zh")

    assert "检索" in text
    assert "must-not-leak" not in text
    assert "secret" not in text
    assert response_language("分析这个结果") == "zh"
    assert response_language("Analyze this result") == "en"


def test_completion_message_projects_summary_bullets_and_real_artifacts():
    text = completion_message(
        {
            "output": {"summary": "已完成真实数据分析。"},
            "completion_bullets": ["生成了结果表", "撰写了分析报告"],
        },
        [
            {"artifact_id": "a-1", "filename": "results.csv"},
            {"artifact_id": "a-2", "filename": "报告.md"},
        ],
        language="zh",
    )

    assert text.startswith("已完成真实数据分析。")
    assert "- 生成了结果表" in text
    assert "[results.csv](/api/artifacts/a-1)" in text
    assert "%E6%8A%A5%E5%91%8A.md" not in text
    assert "](/api/artifacts/a-2)" in text


def test_completion_message_deduplicates_existing_closing_prose():
    text = completion_message(
        {
            "output": {"answer": "The answer is 42."},
            "completion_bullets": ["Computed the answer"],
        },
        previous_text="The answer is 42.\n\nComputed the answer",
        require_fallback=False,
    )

    assert text == ""


def test_completion_message_has_bounded_json_fallback_and_completion_fallback():
    rendered = completion_message(
        {"output": {"metrics": {"accuracy": 0.93}}}, language="en"
    )
    assert '"accuracy": 0.93' in rendered

    assert (
        completion_message(None, language="zh", require_fallback=True)
        == "任务已完成。"
    )
    assert completion_message(None, require_fallback=False) == ""


def test_completion_action_narration_does_not_expose_submit_source():
    text = action_narration(
        CodeCell("python", "host.submit_output({'ok': True}, ['Completed it'])")
    )

    assert text == ""
    assert "submit_output" not in text
