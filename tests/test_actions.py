"""Action-parsing core contracts — openai4s.agent.actions.

Both outer loops (agent/loop.py Agent.run and server/gateway.py
SessionRunner._loop) parse a reply through this single module, so these tests
lock the language whitelist, the one-cell-per-step document-order rule, and
the fence-collision guarantees the dual loop depends on.
"""
from openai4s.agent.actions import (
    MULTI_CELL_NOTE,
    NO_CODE_NUDGE,
    CodeCell,
    count_code_blocks,
    extract_action,
)

_F = "`" * 3


def _cell(info: str, body: str) -> str:
    return f"{_F}{info}\n{body}\n{_F}"


def test_python_infos_and_bare_fence_mean_python():
    for info in ("python", "py", ""):
        action = extract_action(f"prose\n{_cell(info, 'x = 1')}")
        assert action == CodeCell("python", "x = 1\n")


def test_r_fence_is_first_class_and_case_insensitive():
    for info in ("r", "R"):
        action = extract_action(_cell(info, "x <- 1"))
        assert action == CodeCell("r", "x <- 1\n")


def test_document_order_decides_between_languages():
    reply = _cell("r", "a <- 1") + "\nprose\n" + _cell("python", "b = 2")
    assert extract_action(reply).language == "r"
    reply = _cell("python", "b = 2") + "\nprose\n" + _cell("r", "a <- 1")
    assert extract_action(reply).language == "python"


def test_non_action_fences_are_ignored():
    assert extract_action(_cell("json", '{"a": 1}')) is None
    assert extract_action(_cell("tool", '{"name": "list_dir"}')) is None
    assert extract_action("just prose") is None


def test_unclosed_fence_is_never_executable():
    assert extract_action(f"{_F}python\nx = 1\n") is None
    assert extract_action(f"{_F}r\nx <- 1\n") is None


def test_nested_tool_example_stays_inside_the_cell():
    body = 'doc = """\n' + _cell("tool", '{"name": "x"}') + '\n"""'
    action = extract_action(_cell("python", body))
    assert action is not None and action.language == "python"
    assert '{"name": "x"}' in action.code


def test_count_code_blocks_spans_both_languages():
    reply = (
        _cell("python", "a = 1")
        + "\n"
        + _cell("r", "b <- 2")
        + "\n"
        + _cell("json", "{}")
    )
    assert count_code_blocks(reply) == 2
    assert count_code_blocks("prose only") == 0


def test_shared_texts_mention_both_channels():
    assert "```python" in NO_CODE_NUDGE and "```r" in NO_CODE_NUDGE
    assert "only the FIRST" in MULTI_CELL_NOTE
