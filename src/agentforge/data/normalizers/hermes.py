"""Normalizer for NousResearch/hermes-function-calling-v1.

Native format: ShareGPT-style records, `conversations: [{from, value}, ...]`,
where `from` is one of `system` / `human` / `gpt` (and possibly a literal
`tool` role -- unconfirmed, see NOTE below). Function-calling content is
embedded as XML-ish tags inside a turn's `value` string:

  - system turn: `<tools>[...]</tools>` wraps a JSON array of tool defs,
    typically alongside other prose instructing the model how to call them.
  - gpt turn: `<tool_call>{...}</tool_call>` wraps one function-call JSON
    object. A turn may contain more than one such tag (parallel tool calls).
  - tool response: `<tool_response>{...}</tool_response>` wraps one JSON
    object. This dataset is inconsistent about whether a tool response gets
    its own `from: "tool"` conversation entry or is embedded inside a
    `from: "human"` entry instead -- per the task brief we detect this by
    tag presence in `value`, not by trusting the `from` field alone.

NOTE -- assumptions below are UNVERIFIED against the live HF dataset as of
this writing. Confirm all of these against a real `datasets.load_dataset`
pull before relying on this normalizer for a production manifest build:

  1. The exact HF Hub config/file list for hermes-function-calling-v1 is
     not confirmed. `_PLAUSIBLE_CONFIGS` below is a best-guess placeholder
     list (config names that read plausibly off the dataset card), not a
     verified list of what actually exists.
  2. Whether a literal `from: "tool"` role ever appears in `conversations`
     (as opposed to tool responses always being embedded inside a
     `from: "human"` entry) is unconfirmed; both paths are supported here.
  3. The exact key shape inside a `<tool_response>{...}</tool_response>`
     JSON payload (e.g. a `content`/`response`/`result` key holding the
     actual tool output, vs. some other shape) is unconfirmed -- see
     `_tool_response_content` for the lenient, best-effort shape assumed.
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

# Best-guess, UNVERIFIED placeholder config list -- confirm the real
# config/file names against the live `NousResearch/hermes-function-calling-v1`
# dataset on the HF Hub before a real manifest build. These read as plausible
# per the dataset card but have not been checked against the actual repo.
_PLAUSIBLE_CONFIGS = [
    "func-calling",
    "func-calling-singleturn",
    "json-mode-agentic",
    "json-mode-singleturn",
    "glaive-function-calling-5k",
]

_TOOLS_TAG_RE = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_TAG_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _row_id(conversations: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(conversations, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"hermes_{digest}"


def _extract_tools(system_text: str) -> tuple[list[ToolSpec], str]:
    """Return (tool specs found inside <tools>...</tools>, leftover prose).

    Uses `extract_json_objects` (balanced brace-matching) rather than a naive
    regex to parse the JSON payload, since tool parameter schemas nest
    braces (`{"parameters": {"type": "object", ...}}`) and the payload is a
    JSON *array* of objects -- `extract_json_objects` handles this fine
    because it only tracks `{`/`}` depth, treating each top-level object
    inside the array as its own balanced span.
    """
    matches = list(_TOOLS_TAG_RE.finditer(system_text))
    if not matches:
        return [], system_text.strip()
    # Hermes system prompts commonly *describe* the tag before using it, e.g.
    # "...function signatures within <tools></tools> XML tags..." followed
    # later by the real, populated `<tools>[...]</tools>`. A plain first-match
    # search would pick up that empty decoy. Prefer the last match that
    # actually has non-empty content; fall back to the last match overall.
    non_empty = [m for m in matches if m.group(1).strip()]
    match = non_empty[-1] if non_empty else matches[-1]
    objects = extract_json_objects(match.group(1))
    tools: list[ToolSpec] = []
    for obj in objects:
        if isinstance(obj.get("name"), str):
            try:
                tools.append(ToolSpec(function=obj))
            except Exception:  # noqa: BLE001 - malformed tool def, skip just this one
                continue
    prose = (system_text[: match.start()] + system_text[match.end() :]).strip()
    return tools, prose


def _function_call_arguments(payload: dict) -> str:
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return arguments


def _tool_response_content(payload: dict) -> str:
    """Best-effort extraction of a human-readable tool response body.

    UNVERIFIED shape (see module docstring note 3): prefers a `content` /
    `response` / `result` key if present, else dumps the remaining payload
    (minus `name`, which is surfaced separately on the Message).
    """
    for key in ("content", "response", "result"):
        if key in payload:
            value = payload[key]
            return value if isinstance(value, str) else json.dumps(value)
    remainder = {k: v for k, v in payload.items() if k != "name"}
    return json.dumps(remainder) if remainder else json.dumps(payload)


def _parse_tool_responses(
    value: str, open_calls: dict[str, str], *, tag_required: bool = True
) -> tuple[list[Message], bool]:
    """Extract one `Message(role="tool", ...)` per tool-response payload found
    in `value`, linking each back to the most recently opened tool call --
    the same "track open calls" pattern glaive.py uses for its
    `FUNCTION RESPONSE` turns (`next(reversed(open_calls.items()))`, entry
    left in place rather than popped).

    NOTE: unverified against real parallel-tool-call rows -- when a single
    turn carries more than one `<tool_response>` tag, this (like glaive)
    links every response in that turn to the same "most recent" call rather
    than pairing them up distinctly. Getting that pairing right (and which
    order -- FIFO vs LIFO) needs real multi-call Hermes examples to decide
    against; `payload.get("name")` still recovers the correct function name
    per response even when `tool_call_id` linkage is imprecise, and the
    schema's `orphan_tool_response` check only soft-flags mismatches rather
    than dropping the row.

    Returns `(messages, ok)`; `ok=False` means the payload JSON could not be
    parsed at all, which the caller treats as a reason to drop the whole row
    (mirroring how a malformed `<tool_call>` is handled).
    """
    tag_matches = list(_TOOL_RESPONSE_TAG_RE.finditer(value))
    payloads: list[dict] = []
    if tag_matches:
        for m in tag_matches:
            objs = extract_json_objects(m.group(1))
            if not objs:
                return [], False
            payloads.append(objs[0])
    elif not tag_required:
        # from: "tool" turn without the wrapping tag (unconfirmed shape):
        # treat the whole value as the JSON payload directly.
        objs = extract_json_objects(value)
        if not objs:
            return [], False
        payloads.extend(objs)
    else:
        return [], False

    messages: list[Message] = []
    for payload in payloads:
        if open_calls:
            call_id, open_name = next(reversed(open_calls.items()))
        else:
            call_id, open_name = f"call_unmatched_{len(messages)}", "unknown"
        name = payload.get("name") or open_name
        messages.append(
            Message(
                role="tool",
                content=_tool_response_content(payload),
                tool_call_id=call_id,
                name=name,
            )
        )
    return messages, True


class HermesNormalizer(Normalizer):
    source = "hermes"

    def __init__(
        self,
        raw_examples: Iterable[dict[str, Any]] | None = None,
        config_name: str | None = None,
    ) -> None:
        self._raw_examples = raw_examples
        self._config_name = config_name

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if self._raw_examples is not None:
            yield from self._raw_examples
            return
        import datasets  # local import: only required for a real (network) build

        # UNVERIFIED config list -- see module docstring note 1.
        configs = [self._config_name] if self._config_name else _PLAUSIBLE_CONFIGS
        for config in configs:
            ds = datasets.load_dataset(
                "NousResearch/hermes-function-calling-v1", config, split="train"
            )
            for example in ds:
                example = dict(example)
                example["_hermes_config"] = config
                yield example

    def to_canonical(self, raw: dict[str, Any]) -> Row | None:
        conversations = raw.get("conversations")
        if not conversations:
            return None
        config_name = raw.get("_hermes_config", "unknown")

        messages: list[Message] = []
        tools: list[ToolSpec] = []
        open_calls: dict[str, str] = {}  # call_id -> function name, insertion order
        call_counter = 0

        for turn in conversations:
            frm = turn.get("from")
            value = turn.get("value") or ""

            if frm == "system":
                tools, prose = _extract_tools(value)
                if prose:
                    messages.append(Message(role="system", content=prose))
                continue

            if frm == "human":
                if "<tool_response>" in value:
                    tool_msgs, ok = _parse_tool_responses(value, open_calls)
                    if not ok:
                        return None
                    messages.extend(tool_msgs)
                else:
                    messages.append(Message(role="user", content=value))
                continue

            if frm == "tool":
                # Unconfirmed: a literal "tool" role may or may not wrap its
                # payload in a <tool_response> tag -- support both.
                tool_msgs, ok = _parse_tool_responses(value, open_calls, tag_required=False)
                if not ok:
                    return None
                messages.extend(tool_msgs)
                continue

            if frm == "gpt":
                call_tags = list(_TOOL_CALL_TAG_RE.finditer(value))
                if not call_tags:
                    if value.strip():
                        messages.append(Message(role="assistant", content=value.strip()))
                    continue
                tool_calls: list[ToolCall] = []
                for tag_match in call_tags:
                    objs = extract_json_objects(tag_match.group(1))
                    if not objs:
                        # Malformed embedded JSON: drop the whole row rather
                        # than guess at a broken tool call.
                        return None
                    payload = objs[0]
                    name = payload.get("name")
                    if not name:
                        return None
                    call_id = f"call_{call_counter}"
                    call_counter += 1
                    open_calls[call_id] = name
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            function=FunctionCall(name=name, arguments=_function_call_arguments(payload)),
                        )
                    )
                prose = _TOOL_CALL_TAG_RE.sub("", value).strip() or None
                messages.append(Message(role="assistant", content=prose, tool_calls=tool_calls))
                continue

            # Unknown/unhandled speaker role: skip the turn rather than
            # crash or drop the whole row over one unexpected `from` value.

        if not messages or messages[0].role not in ("system", "user"):
            return None

        return Row(
            id=_row_id(conversations),
            source=self.source,
            messages=messages,
            tools=tools,
            meta={"hermes_config": config_name},
        )
