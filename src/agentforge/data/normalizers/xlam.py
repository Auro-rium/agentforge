"""Normalizer for Salesforce/xlam-function-calling-60k.

Native format: flat columns per row (no multi-turn structure to parse).
  - `query`: str, the user's request.
  - `tools`: JSON-encoded string of a list of tool definitions, each like
    `{"name": ..., "description": ..., "parameters": {...}}`.
  - `answers`: JSON-encoded string of a list of `{"name": ..., "arguments": {...}}`
    dicts -- the expected tool call(s) for `query`.

This is the simplest of the five normalizers: single-turn, no ReAct-style text
parsing required, just two JSON blobs to unpack per row.

Note: `Salesforce/xlam-function-calling-60k` is a *gated* dataset on the Hub.
Building the manifest for real requires accepting its terms and setting
`HF_TOKEN`; see `_GATED_DATASET_HELP` below for the error raised otherwise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

from agentforge.data.normalizers.base import Normalizer
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall, ToolSpec

try:
    from huggingface_hub.errors import GatedRepoError as _GatedRepoError
except ImportError:  # pragma: no cover - older huggingface_hub layouts
    _GatedRepoError = None

_GATED_DATASET_HELP = (
    "Salesforce/xlam-function-calling-60k is a gated HF dataset. Accept its terms at "
    "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k and set "
    "HF_TOKEN before building the manifest."
)

_GATED_ERROR_MARKERS = (
    "gated",
    "401",
    "403",
    "unauthorized",
    "restricted",
    "authenticated",
)


def _looks_like_gated_access_error(exc: Exception) -> bool:
    if _GatedRepoError is not None and isinstance(exc, _GatedRepoError):
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _GATED_ERROR_MARKERS)


def _row_id(raw_index: int, query: str) -> str:
    digest = hashlib.sha256(f"xlam:{raw_index}:{query}".encode()).hexdigest()[:16]
    return f"xlam_{digest}"


class XlamNormalizer(Normalizer):
    source = "xlam"

    def __init__(self, raw_examples: Iterable[dict[str, Any]] | None = None) -> None:
        self._raw_examples = raw_examples

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if self._raw_examples is not None:
            yield from self._raw_examples
            return
        import datasets  # local import: only required for a real (network) build

        try:
            ds = datasets.load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        except Exception as exc:  # noqa: BLE001 - re-raise as an actionable error below
            if _looks_like_gated_access_error(exc):
                raise RuntimeError(_GATED_DATASET_HELP) from exc
            raise
        yield from ds

    def to_canonical(self, raw: dict[str, Any]) -> Row | None:
        query = raw.get("query")
        tools_str = raw.get("tools")
        answers_str = raw.get("answers")

        if not isinstance(query, str) or not query.strip():
            return None
        if tools_str is None or answers_str is None:
            return None

        tools_raw = _parse_json_field(tools_str)
        if not isinstance(tools_raw, list):
            return None
        answers_raw = _parse_json_field(answers_str)
        if not isinstance(answers_raw, list):
            return None

        if not answers_raw:
            # No expected tool call for this query. An assistant turn with
            # content=None and tool_calls=[] would carry no learnable signal
            # (nothing for the model to imitate), so drop rather than emit it.
            return None

        tools: list[ToolSpec] = []
        for t in tools_raw:
            if not isinstance(t, dict):
                return None
            try:
                tools.append(ToolSpec(function=t))
            except Exception:  # noqa: BLE001 - malformed tool def -> drop the row
                return None

        tool_calls: list[ToolCall] = []
        for i, ans in enumerate(answers_raw):
            if not isinstance(ans, dict) or not ans.get("name"):
                return None
            try:
                arguments_json = json.dumps(ans.get("arguments", {}))
            except TypeError:
                return None
            tool_calls.append(
                ToolCall(
                    id=f"call_{i}",
                    function=FunctionCall(name=ans["name"], arguments=arguments_json),
                )
            )

        messages = [
            Message(role="user", content=query),
            Message(role="assistant", content=None, tool_calls=tool_calls),
        ]

        return Row(
            id=_row_id(hash(query) & 0xFFFFFFFF, query),
            source=self.source,
            messages=messages,
            tools=tools,
        )


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON-encoded column value; return None on any malformed input."""
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
