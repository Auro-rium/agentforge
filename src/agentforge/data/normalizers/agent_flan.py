"""Normalizer for internlm/Agent-FLAN.

Native format (per HF dataset card, verified via research -- reconfirm exact
split names against the live dataset at real-build time, this is not a
guaranteed-current snapshot): two columns per row, `id: str` and
`conversation: [{role, content, loss}]`. `role` is limited to
`system`/`user`/`assistant` -- there is no native `tool` role and no
structured tool_calls anywhere in the raw data; everything is free text.
`loss` is a per-turn training-mask flag from the original Agent-FLAN paper
and has no slot in our canonical schema, so it's intentionally dropped here.

Splits on the hub (research-time snapshot):
  - agent_instruct_react        (ReAct trajectories, kept by default)
  - agent_instruct_tflan        (reformulated instruction data, no raw
                                  Thought/Action/Observation structure)
  - toolbench_instruct_j1s1_3k  (reformulated instruction data)
  - toolbench_negative          (reformulated instruction data)
  - toolbench_react_10p         (ReAct trajectories, kept by default)
  - toolbench_tflan_60p_r10r5u7 (reformulated instruction data)
  - toolbench_tflan_cot_30p     (reformulated instruction data)

Only the two ReAct-bearing splits are loaded by default -- the `*_tflan*`/
`*_instruct_j1s1*` splits are already flattened into instruction-style data
without a parseable Thought/Action/Observation trajectory, so they're left
loadable via the `splits` constructor override for a later ablation but
excluded by default.

This is the headline multi-turn ReAct dataset for the project (the other
four source datasets teach single-turn schema-grounded argument
correctness; this is the only one teaching genuine multi-turn
Thought/Action/Observation reasoning loops), so parsing fidelity here
matters more than in any other normalizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from agentforge.data.normalizers.base import Normalizer
from agentforge.data.react_parsing import (
    is_observation_turn,
    parse_react_turn,
    react_action_input_to_arguments_json,
    strip_observation_prefix,
)
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall

DEFAULT_REACT_SPLITS: tuple[str, ...] = ("agent_instruct_react", "toolbench_react_10p")

_SYNTHESIZE_TOOL_SCHEMAS_MSG = (
    "synthesize_tool_schemas=True is not implemented: no per-row tool schema "
    "catalog exists in Agent-FLAN, and synthesizing fake schemas risks teaching "
    "the model to ignore the tools argument. This is a documented future toggle, "
    "not something to build without a deliberate ablation plan."
)


class AgentFlanNormalizer(Normalizer):
    source = "agent_flan"

    def __init__(
        self,
        raw_examples: Iterable[dict[str, Any]] | None = None,
        splits: tuple[str, ...] = DEFAULT_REACT_SPLITS,
        synthesize_tool_schemas: bool = False,
    ) -> None:
        if synthesize_tool_schemas:
            # Raised eagerly at construction time (not deferred to first use)
            # so a misconfigured build fails fast, before any dataset loading
            # or normalization work happens.
            raise NotImplementedError(_SYNTHESIZE_TOOL_SCHEMAS_MSG)
        self._raw_examples = raw_examples
        self.splits = splits
        self.synthesize_tool_schemas = synthesize_tool_schemas

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if self._raw_examples is not None:
            yield from self._raw_examples
            return
        import datasets  # local import: only required for a real (network) build

        for split in self.splits:
            ds = datasets.load_dataset("internlm/Agent-FLAN", split=split)
            for example in ds:
                yield {**example, "_split": split}

    def to_canonical(self, raw: dict[str, Any]) -> Row | None:
        conversation = raw.get("conversation")
        if not conversation:
            return None

        split = raw.get("_split", "")
        meta: dict[str, Any] = {"source_split": split}

        messages: list[Message] = []
        open_calls: dict[str, str] = {}  # call_id -> action/function name, in emission order
        call_counter = 0

        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content") or ""

            if role == "system":
                messages.append(Message(role="system", content=content))

            elif role == "user":
                if is_observation_turn(content):
                    stripped = strip_observation_prefix(content).strip() or None
                    if open_calls:
                        call_id, name = next(reversed(open_calls.items()))
                    else:
                        # Malformed trajectory: an Observation with no
                        # currently-open call. Still emit the tool message --
                        # the schema's soft orphan_tool_response tagging
                        # handles this, no need to special-case it here.
                        call_id, name = f"call_{call_counter}", "unknown"
                    messages.append(
                        Message(role="tool", content=stripped, tool_call_id=call_id, name=name)
                    )
                else:
                    messages.append(Message(role="user", content=content))

            elif role == "assistant":
                parsed = parse_react_turn(content)
                if parsed is not None:
                    arguments, args_raw = react_action_input_to_arguments_json(
                        parsed.action_input_raw
                    )
                    call_id = f"call_{call_counter}"
                    call_counter += 1
                    open_calls[call_id] = parsed.action
                    if args_raw:
                        meta["args_raw"] = True
                    messages.append(
                        Message(
                            role="assistant",
                            content=parsed.thought or None,
                            tool_calls=[
                                ToolCall(
                                    id=call_id,
                                    function=FunctionCall(name=parsed.action, arguments=arguments),
                                )
                            ],
                        )
                    )
                else:
                    # Row-level flag, not per-turn: every well-formed ReAct
                    # trajectory ends in a terminal free-text answer turn
                    # (no "Action:" line left to parse), so this will be set
                    # True on nearly all rows, including fully-structured
                    # multi-cycle ones. It means "at least one assistant
                    # turn in this row wasn't structured," not "this row has
                    # no tool calls" -- don't filter on it alone downstream.
                    meta["parsed_from_text"] = True
                    messages.append(Message(role="assistant", content=content or None))

            # Any other role in the raw data is not part of the documented
            # system/user/assistant-only native format; skip it rather than
            # guess at a mapping.

        if not messages or messages[0].role not in ("system", "user"):
            return None

        raw_id = raw.get("id", "")
        if raw_id:
            row_id = f"agent_flan_{split}__{raw_id}"
        else:
            fallback_hash = hash(str(conversation)) & 0xFFFFFFFF
            row_id = f"agent_flan_{split}__{fallback_hash}"

        return Row(
            id=row_id,
            source=self.source,
            messages=messages,
            tools=[],
            meta=meta,
        )
