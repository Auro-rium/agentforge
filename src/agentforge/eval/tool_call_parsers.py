"""Model-family-aware parsing of tool calls out of raw generated text.

`google/gemma-4-12B-it` is the project's target model, and per its
documentation ships native `<|tool_call>`/`<|tool_response>`/`<|tool|>`
special tokens -- but the exact generation-time framing (e.g. whether
`<|tool_call>` is immediately followed by a bare JSON object, whether
multiple calls are wrapped in a list, whether there's an explicit closing
marker) was not independently confirmed against real model output during
this build (no local GPU to generate from the real 12B model). The parser
below is written defensively -- balanced-brace JSON extraction after each
`<|tool_call>` marker, repeated for multiple calls -- and should be spot
-checked against real Gemma 4 generations at the first real eval run.

A generic JSON-fence / XML-tag fallback (covers Qwen-style `<tool_call>...
</tool_call>`, and a bare ```json fenced block, or a bare top-level JSON
object/list) is also provided for other model families and as a safety net
if the Gemma-specific parser finds nothing.
"""

from __future__ import annotations

import contextlib
import json
import re

from agentforge.data.react_parsing import extract_json_objects, find_balanced_brace_span

GEMMA_TOOL_CALL_MARKER = "<|tool_call>"
QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_gemma4_tool_calls(text: str) -> list[dict]:
    """Extract tool calls following `<|tool_call>` markers.

    Only accepts a JSON object that starts at (or right after skipping
    whitespace from) the marker's own position -- NOT "the nearest object
    anywhere later in the text," which would misattribute a later marker's
    object to an earlier marker that has no JSON directly following it (a
    real bug caught by `test_marker_with_no_json_then_marker_with_json`
    during development: using whole-string `extract_json_objects` on the
    remainder-after-marker over-matched into subsequent markers' objects).
    `find_balanced_brace_span` anchors to one exact starting index instead.
    """
    calls: list[dict] = []
    idx = 0
    while True:
        marker_idx = text.find(GEMMA_TOOL_CALL_MARKER, idx)
        if marker_idx == -1:
            break
        after_marker = marker_idx + len(GEMMA_TOOL_CALL_MARKER)
        brace_start = after_marker
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1
        span = find_balanced_brace_span(text, brace_start)
        if span is not None:
            with contextlib.suppress(json.JSONDecodeError):
                calls.append(json.loads(span))
        idx = after_marker
    return calls


def parse_generic_tool_calls(text: str) -> list[dict]:
    """Fallback parser for non-Gemma model families / as a safety net:
    Qwen-style `<tool_call>...</tool_call>` tags, ```json fenced blocks, or
    a bare top-level JSON object/list anywhere in the text.
    """
    calls: list[dict] = []

    for match in QWEN_TOOL_CALL_RE.finditer(text):
        objects = extract_json_objects(match.group(1))
        calls.extend(objects)
    if calls:
        return calls

    for match in JSON_FENCE_RE.finditer(text):
        objects = extract_json_objects(match.group(1))
        calls.extend(objects)
    if calls:
        return calls

    return extract_json_objects(text)


def parse_tool_calls(text: str, *, model_family: str = "gemma4") -> list[dict]:
    """Dispatch to the appropriate parser by model family, falling back to
    the generic parser if the family-specific one finds nothing (covers the
    case where the real Gemma 4 output format differs from what's assumed
    above).
    """
    if model_family == "gemma4":
        calls = parse_gemma4_tool_calls(text)
        if calls:
            return calls
    return parse_generic_tool_calls(text)
