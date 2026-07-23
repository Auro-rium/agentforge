"""Normalizer for glaiveai/glaive-function-calling-v2.

Native format: two string columns per row.
  - `system`: prose plus embedded JSON function definitions (one or more
    `{"name": ..., "description": ..., "parameters": {...}}` objects).
  - `chat`: a single string with turns marked by literal `USER:`,
    `ASSISTANT:`, `FUNCTION RESPONSE:` prefixes, and `<functioncall>
    {json}` inline within ASSISTANT turns.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from agentforge.data.normalizers.base import Normalizer
from agentforge.data.react_parsing import extract_json_objects, find_balanced_brace_span
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall, ToolSpec

_TURN_SPLIT_RE = re.compile(r"\n(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
_FUNCTIONCALL_MARKER_RE = re.compile(r"<functioncall>\s*")
_SINGLE_QUOTED_JSON_RE = re.compile(r"'(\{.*\})'", re.DOTALL)


def _parse_functioncall_payload(candidate: str) -> dict | None:
    """Parse a `<functioncall>` payload, repairing glaive's known quirk where
    `arguments` is wrapped in single quotes around an already-JSON string
    (making the outer object invalid JSON as-is)."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    repaired = _SINGLE_QUOTED_JSON_RE.sub(r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _row_id(raw_index: int, chat: str) -> str:
    digest = hashlib.sha256(f"glaive:{raw_index}:{chat}".encode()).hexdigest()[:16]
    return f"glaive_{digest}"


def _extract_tools(system_text: str) -> tuple[list[ToolSpec], str]:
    """Return (tool specs found in `system_text`, leftover non-JSON prose)."""
    objects = extract_json_objects(system_text)
    tools: list[ToolSpec] = []
    for obj in objects:
        if isinstance(obj.get("name"), str):
            try:
                tools.append(ToolSpec(function=obj))
            except Exception:  # noqa: BLE001 - malformed tool def, skip just this one
                continue
    # Strip the JSON blocks out, leaving only prose, for an optional synthetic
    # system message. This is best-effort cosmetic cleanup, not load-bearing.
    prose = re.sub(r"\{.*?\}(?=\s*\{|\s*$)", "", system_text, flags=re.DOTALL).strip()
    return tools, prose


def _split_turns(chat: str) -> list[tuple[str, str]]:
    """Split the `chat` string into (speaker, text) turns in order."""
    chat = chat.strip()
    if not chat:
        return []
    parts = _TURN_SPLIT_RE.split("\n" + chat)
    # parts[0] is whatever precedes the first marker (should be empty after the
    # leading "\n" we prepended); then alternating (speaker, text) pairs.
    turns: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        speaker = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append((speaker, text))
    return turns


class GlaiveNormalizer(Normalizer):
    source = "glaive"

    def __init__(self, raw_examples: Iterable[dict[str, Any]] | None = None) -> None:
        self._raw_examples = raw_examples

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if self._raw_examples is not None:
            yield from self._raw_examples
            return
        import datasets  # local import: only required for a real (network) build

        ds = datasets.load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        yield from ds

    def to_canonical(self, raw: dict[str, Any]) -> Row | None:
        system_text = raw.get("system") or ""
        chat = raw.get("chat") or ""
        if not chat.strip():
            return None

        tools, prose = _extract_tools(system_text)

        messages: list[Message] = []
        if prose:
            messages.append(Message(role="system", content=prose))

        open_calls: dict[str, str] = {}  # call_id -> function name, in emission order
        call_counter = 0

        for speaker, text in _split_turns(chat):
            if speaker == "USER":
                messages.append(Message(role="user", content=text))
            elif speaker == "ASSISTANT":
                marker_match = _FUNCTIONCALL_MARKER_RE.search(text)
                if marker_match is None:
                    if text:
                        messages.append(Message(role="assistant", content=text))
                    continue
                brace_start = marker_match.end()
                candidate = find_balanced_brace_span(text, brace_start)
                if candidate is None:
                    return None
                payload = _parse_functioncall_payload(candidate)
                if payload is None:
                    # Malformed embedded JSON: drop the whole row rather than
                    # guess at a broken tool call.
                    return None
                name = payload.get("name")
                if not name:
                    return None
                call_id = f"call_{call_counter}"
                call_counter += 1
                open_calls[call_id] = name
                arguments = payload.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                full_call_span = text[marker_match.start() : brace_start + len(candidate)]
                remaining_prose = text.replace(full_call_span, "").strip() or None
                messages.append(
                    Message(
                        role="assistant",
                        content=remaining_prose,
                        tool_calls=[
                            ToolCall(
                                id=call_id, function=FunctionCall(name=name, arguments=arguments)
                            )
                        ],
                    )
                )
            elif speaker == "FUNCTION RESPONSE":
                if open_calls:
                    call_id, name = next(reversed(open_calls.items()))
                else:
                    call_id, name = f"call_{call_counter}", "unknown"
                messages.append(
                    Message(role="tool", content=text, tool_call_id=call_id, name=name)
                )

        if not messages or messages[0].role not in ("system", "user"):
            return None

        return Row(
            id=_row_id(hash(chat) & 0xFFFFFFFF, chat),
            source=self.source,
            messages=messages,
            tools=tools,
        )
