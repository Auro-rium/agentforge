"""Parsing helpers shared across normalizers (and eval/tool_call_parsers.py).

`extract_json_objects` and the ReAct regex are used by more than one
normalizer (glaive/toolace share the brace-matching extractor; toolace/
agent_flan share the "next turn is actually an observation" remap), so they
live here rather than being duplicated per-normalizer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Thought: ... \n Action: <name> \n Action Input: <anything>
# DOTALL so Thought's content can span multiple lines; the Action/Action Input
# lines are anchored so a stray "Action:"-looking word inside prose doesn't
# false-match mid-Thought. Action Input is captured up to the next
# Thought:/Action:/Observation: marker (or end of string) rather than
# requiring a `{...}` wrapper -- a genuinely brace-less Action Input (e.g.
# `Action Input: city=Paris`) is real, malformed ToolBench-derived data, and
# should still reach the lenient JSON-repair fallback below rather than
# silently falling through to the parsed_from_text branch untried. An
# earlier version of this regex required a leading `{`, which meant only
# brace-wrapped-but-invalid JSON (e.g. `{city: Paris}`) ever reached the
# fallback -- caught during agent_flan.py normalizer development.
REACT_TURN_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*\n"
    r"Action:\s*(?P<action>\S+)\s*\n"
    r"Action Input:\s*(?P<action_input>.*?)"
    r"(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
    re.DOTALL,
)

OBSERVATION_PREFIX_RE = re.compile(r"^\s*Observation:\s*", re.IGNORECASE)


def find_balanced_brace_span(text: str, start: int) -> str | None:
    """From `text[start]` (expected to be '{'), return the balanced `{...}`
    span, tracking only `"`-delimited strings (JSON semantics) so an inner
    `'`-wrapped JSON blob -- e.g. glaive's real, documented quirk of
    embedding `"arguments": '{"city": "Paris"}'` -- doesn't prematurely
    close the span at its first inner `}`. Returns None if `text[start]`
    isn't `{` or no balancing `}` is found.

    Unlike `extract_json_objects` below (which scans an entire string for
    *all* top-level objects anywhere in it), this anchors to one specific
    starting position -- the right tool when you need "the object starting
    exactly here," not "any object somewhere in this text."
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_objects(text: str) -> list[dict]:
    """Extract top-level JSON objects embedded in `text` via brace matching.

    Naive regex (e.g. `\\{.*\\}`) breaks on nested braces, which tool
    parameter schemas (`{"parameters": {"type": "object", ...}}`) always
    have. This scans for balanced `{...}` spans instead, greedily matching
    from each unescaped `{` to its balancing `}`, and tries to json.loads
    each candidate span -- non-JSON spans (stray braces in prose) are
    skipped rather than raising.
    """
    objects: list[dict] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":  # noqa: SIM102 -- depth -= 1 must run before the depth==0 check, can't merge
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(parsed, dict):
                            objects.append(parsed)
                    start = None
    return objects


@dataclass
class ParsedReactTurn:
    thought: str
    action: str
    action_input_raw: str
    args_raw: bool  # True if action_input_raw wasn't valid JSON (lenient fallback used)


def parse_react_turn(content: str) -> ParsedReactTurn | None:
    """Parse a single Thought/Action/Action Input block from free-text content.

    Returns None on no match (caller should keep the turn as plain content,
    tagged meta.parsed_from_text=True, never drop the row for a parse miss).
    """
    match = REACT_TURN_RE.search(content)
    if match is None:
        return None
    return ParsedReactTurn(
        thought=match.group("thought").strip(),
        action=match.group("action").strip(),
        action_input_raw=match.group("action_input").strip(),
        args_raw=False,
    )


def react_action_input_to_arguments_json(action_input_raw: str) -> tuple[str, bool]:
    """Return (arguments_json_string, args_raw) for a parsed Action Input.

    Valid JSON passes through unchanged. Invalid JSON (common in ToolBench-
    derived Agent-FLAN rows) gets wrapped as {"input": <raw text>} rather than
    dropping the row -- tagged args_raw=True so it's distinguishable later.
    """
    try:
        json.loads(action_input_raw)
    except json.JSONDecodeError:
        return json.dumps({"input": action_input_raw}), True
    return action_input_raw, False


def is_observation_turn(content: str) -> bool:
    return bool(OBSERVATION_PREFIX_RE.match(content))


def strip_observation_prefix(content: str) -> str:
    return OBSERVATION_PREFIX_RE.sub("", content, count=1)
