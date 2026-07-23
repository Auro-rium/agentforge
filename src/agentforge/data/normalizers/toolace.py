"""Normalizer for Team-ACE/ToolACE.

Native format: two columns per row.
  - `system`: prose plus embedded JSON tool-definition objects, same shape as
    glaive's `system` column (reuse `extract_json_objects`).
  - `conversations`: a ShareGPT-format list of `{"from": "human"|"gpt",
    "value": str}` turns. There is no native tool/observation role in this
    dataset -- every turn is either `human` or `gpt`.

Mapping logic:
  - `from` -> role: `human` -> `user`, `gpt` -> `assistant`.
  - A `gpt` turn's `value` is attempted as `json.loads(value)`. If it parses
    as a non-empty list of `{"name": ..., "arguments": ...}`-shaped dicts,
    it's treated as a structured tool call: each list item becomes a
    `ToolCall` with a synthetic `call_{idx}` id, and the message's `content`
    is `None`. Otherwise (plain prose, or JSON that parses to something else
    -- a dict, a number, an empty list, a list of non-tool-call items) the
    original string is kept verbatim as ordinary assistant `content`, with no
    `tool_calls`.
  - See `_should_remap_to_tool_role` below for the (unverified) heuristic
    that turns a `human` turn immediately following a tool-calling
    `assistant` turn into a synthetic `role="tool"` response instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from agentforge.data.normalizers.base import Normalizer
from agentforge.data.react_parsing import extract_json_objects
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall, ToolSpec


def _row_id(raw_index: int, conversations: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(f"toolace:{raw_index}:{conversations!r}".encode()).hexdigest()[:16]
    return f"toolace_{digest}"


def _extract_tools(system_text: str) -> tuple[list[ToolSpec], str]:
    """Return (tool specs found in `system_text`, leftover non-JSON prose).

    Mirrors glaive._extract_tools: ToolACE's `system` column has the same
    "prose plus embedded JSON tool defs" shape as glaive's.
    """
    objects = extract_json_objects(system_text)
    tools: list[ToolSpec] = []
    for obj in objects:
        if isinstance(obj.get("name"), str):
            try:
                tools.append(ToolSpec(function=obj))
            except Exception:  # noqa: BLE001 - malformed tool def, skip just this one
                continue
    prose = re.sub(r"\{.*?\}(?=\s*\{|\s*$)", "", system_text, flags=re.DOTALL).strip()
    return tools, prose


def _parse_structured_tool_calls(value: str) -> list[dict] | None:
    """If `value` json-parses as a non-empty list of
    `{"name": ..., "arguments": ...}`-shaped dicts, return that list.
    Otherwise return None -- caller should fall back to treating `value` as
    plain prose content, verbatim, unmodified.

    Explicitly excludes: invalid JSON, a bare dict, a number/string/bool, an
    empty list, and a list whose items aren't all tool-call-shaped dicts
    (e.g. a list of strings). Any of those is prose, not a tool call.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    is_tool_call_shaped = all(
        isinstance(item, dict) and "name" in item and "arguments" in item for item in parsed
    )
    if not is_tool_call_shaped:
        return None
    return parsed


def _should_remap_to_tool_role(messages: list[Message]) -> bool:
    """Decide whether the next `human` turn should be remapped to
    `role="tool"` instead of `role="user"`.

    UNVERIFIED HEURISTIC -- flagged explicitly for later verification against
    real ToolACE data. The rule implemented here: if the most recently
    appended message is an `assistant` turn carrying `tool_calls`, the very
    next `human` turn is assumed to be a simulated environment/tool
    observation being fed back into the synthetic multi-turn dialogue,
    rather than a genuine new user question. This is a plausible pattern for
    ToolACE's synthetic-dialogue construction process, but it has NOT been
    confirmed against real ToolACE rows. Before relying on this in a real
    manifest build, pull ~10 real rows from Team-ACE/ToolACE and check
    whether "human turn right after a tool call" is actually always a tool
    response, or whether it sometimes legitimately is a genuine multi-turn
    user follow-up question that would be misclassified as a tool response
    by this rule.
    """
    if not messages:
        return False
    last = messages[-1]
    return last.role == "assistant" and bool(last.tool_calls)


class ToolACENormalizer(Normalizer):
    source = "toolace"

    def __init__(self, raw_examples: Iterable[dict[str, Any]] | None = None) -> None:
        self._raw_examples = raw_examples

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if self._raw_examples is not None:
            yield from self._raw_examples
            return
        import datasets  # local import: only required for a real (network) build

        ds = datasets.load_dataset("Team-ACE/ToolACE", split="train")
        yield from ds

    def to_canonical(self, raw: dict[str, Any]) -> Row | None:
        system_text = raw.get("system") or ""
        conversations = raw.get("conversations") or []
        if not conversations:
            return None

        tools, prose = _extract_tools(system_text)

        messages: list[Message] = []
        if prose:
            messages.append(Message(role="system", content=prose))

        open_calls: dict[str, str] = {}  # call_id -> function name, in emission order
        call_counter = 0

        for turn in conversations:
            speaker = turn.get("from")
            value = turn.get("value")
            if value is None:
                continue

            if speaker == "gpt":
                structured = _parse_structured_tool_calls(value)
                if structured is None:
                    if value:
                        messages.append(Message(role="assistant", content=value))
                    continue

                tool_calls: list[ToolCall] = []
                for item in structured:
                    name = item.get("name")
                    if not name:
                        continue
                    arguments = item.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    call_id = f"call_{call_counter}"
                    call_counter += 1
                    open_calls[call_id] = name
                    tool_calls.append(
                        ToolCall(id=call_id, function=FunctionCall(name=name, arguments=arguments))
                    )

                if not tool_calls:
                    # Every item in the list was malformed (missing name) --
                    # nothing usable came out of this turn, drop just the turn.
                    continue

                messages.append(Message(role="assistant", content=None, tool_calls=tool_calls))

            elif speaker == "human":
                if _should_remap_to_tool_role(messages):
                    if open_calls:
                        call_id, name = next(reversed(open_calls.items()))
                    else:
                        call_id, name = f"call_{call_counter}", "unknown"
                    messages.append(
                        Message(role="tool", content=value, tool_call_id=call_id, name=name)
                    )
                else:
                    messages.append(Message(role="user", content=value))
            # Any other `from` value (e.g. a leaked "system" entry) is
            # skipped rather than raising -- the task states ToolACE only
            # has human/gpt turns, but real ShareGPT-format data sometimes
            # surprises you, so this is a defensive no-op, not a real path.

        if not messages or messages[0].role not in ("system", "user"):
            return None

        conv_hash = hash(json.dumps(conversations, sort_keys=True, default=str)) & 0xFFFFFFFF
        return Row(
            id=_row_id(conv_hash, conversations),
            source=self.source,
            messages=messages,
            tools=tools,
        )
