import json

import pytest

from agentforge.data.normalizers.agent_flan import DEFAULT_REACT_SPLITS, AgentFlanNormalizer

SYSTEM_PROMPT = "You are an agent with access to tools: get_weather, get_time."

# --- (a) clean multi-turn trajectory: two full Thought/Action/Action Input/
# Observation cycles followed by a final free-text answer. ---
CLEAN_TRAJECTORY = {
    "id": "traj_001",
    "_split": "agent_instruct_react",
    "conversation": [
        {"role": "system", "content": SYSTEM_PROMPT, "loss": 0},
        {"role": "user", "content": "What's the weather and time in Paris?", "loss": 0},
        {
            "role": "assistant",
            "content": (
                "Thought: I need the weather first.\n"
                "Action: get_weather\n"
                'Action Input: {"city": "Paris"}'
            ),
            "loss": 1,
        },
        {"role": "user", "content": 'Observation: {"temperature": "18C"}', "loss": 0},
        {
            "role": "assistant",
            "content": (
                "Thought: Now I need the time.\n"
                "Action: get_time\n"
                'Action Input: {"city": "Paris"}'
            ),
            "loss": 1,
        },
        {"role": "user", "content": 'Observation: {"time": "14:00"}', "loss": 0},
        {
            "role": "assistant",
            "content": (
                "Thought: I have both pieces of information now.\n"
                "Final Answer: It's 18C and 14:00 in Paris."
            ),
            "loss": 1,
        },
    ],
}

# --- (b) Action Input has braces but isn't valid JSON -- exercises the
# lenient {"input": ...} fallback in react_action_input_to_arguments_json.
#
# NOTE on a react_parsing.py discrepancy found while writing this test:
# REACT_TURN_RE hard-requires the Action Input to be wrapped in `{...}`
# (`Action Input:\s*(?P<action_input>\{.*\})`). A brace-less malformed input
# like `Action Input: city=Paris` (the example given in this normalizer's
# task spec) does NOT match REACT_TURN_RE at all -- parse_react_turn returns
# None for it, so it falls into the parsed_from_text branch, never reaching
# react_action_input_to_arguments_json's lenient-JSON fallback. That fallback
# is only reachable for inputs that ARE brace-wrapped but not valid JSON
# (e.g. unquoted keys/values, single quotes). Used such an input below to
# actually exercise the lenient path. Flagged for the react_parsing.py owner
# -- not fixed here per task constraints.
MALFORMED_JSON_TRAJECTORY = {
    "id": "traj_002",
    "_split": "toolbench_react_10p",
    "conversation": [
        {"role": "user", "content": "weather in paris?", "loss": 0},
        {
            "role": "assistant",
            "content": (
                "Thought: I should check the weather.\n"
                "Action: get_weather\n"
                "Action Input: {city: Paris}"
            ),
            "loss": 1,
        },
    ],
}

# --- (c) assistant turn with plain conversational content, no Action: line
# at all. ---
PLAIN_TEXT_TRAJECTORY = {
    "id": "traj_003",
    "_split": "agent_instruct_react",
    "conversation": [
        {"role": "user", "content": "Hey, can you help me?", "loss": 0},
        {"role": "assistant", "content": "Sure, I can help with that. What do you need?", "loss": 1},
    ],
}

# --- (d) a user turn that does not start with "Observation:" -- must stay
# role="user". ---
ORDINARY_USER_TRAJECTORY = {
    "id": "traj_004",
    "_split": "agent_instruct_react",
    "conversation": [
        {"role": "user", "content": "What's the capital of France?", "loss": 0},
        {"role": "assistant", "content": "The capital of France is Paris.", "loss": 1},
        {"role": "user", "content": "Thanks, and what about Germany?", "loss": 0},
    ],
}


class TestAgentFlanCleanReactTrajectory:
    def test_emits_one_row(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0
        assert len(rows) == 1

    def test_role_sequence(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        roles = [m.role for m in rows[0].messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]

    def test_first_call_linked_to_first_observation(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        msgs = rows[0].messages
        first_call_msg = msgs[2]
        first_obs_msg = msgs[3]
        assert first_call_msg.tool_calls is not None
        assert len(first_call_msg.tool_calls) == 1
        call = first_call_msg.tool_calls[0]
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "Paris"}
        assert first_obs_msg.role == "tool"
        assert first_obs_msg.tool_call_id == call.id
        assert first_obs_msg.name == "get_weather"
        assert first_obs_msg.content == '{"temperature": "18C"}'

    def test_second_call_linked_to_second_observation(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        msgs = rows[0].messages
        second_call_msg = msgs[4]
        second_obs_msg = msgs[5]
        assert second_call_msg.tool_calls is not None
        call = second_call_msg.tool_calls[0]
        assert call.function.name == "get_time"
        assert json.loads(call.function.arguments) == {"city": "Paris"}
        assert second_obs_msg.role == "tool"
        assert second_obs_msg.tool_call_id == call.id
        assert second_obs_msg.name == "get_time"
        assert second_obs_msg.content == '{"time": "14:00"}'

    def test_call_ids_are_distinct(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        msgs = rows[0].messages
        first_call_id = msgs[2].tool_calls[0].id
        second_call_id = msgs[4].tool_calls[0].id
        assert first_call_id != second_call_id

    def test_thought_text_becomes_assistant_content(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        msgs = rows[0].messages
        assert msgs[2].content == "I need the weather first."
        assert msgs[4].content == "Now I need the time."

    def test_final_unparsed_turn_kept_as_plain_text(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        final_msg = rows[0].messages[-1]
        assert final_msg.role == "assistant"
        assert final_msg.tool_calls is None
        assert "Final Answer" in final_msg.content

    def test_no_orphan_tool_response_flagged(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta.get("orphan_tool_response") is not True

    def test_parsed_from_text_flag_on_a_row_level_covers_the_terminal_turn(self) -> None:
        # NOTE (semantic caveat, documented deliberately, not accidental):
        # `meta["parsed_from_text"]` is row-level, not per-turn. Every
        # well-formed ReAct trajectory ends in a terminal free-text answer
        # turn (no "Action:" line -- there's nothing left to call), so that
        # final turn always fails parse_react_turn and sets this flag, even
        # on an otherwise fully-structured multi-cycle trajectory like this
        # one. This is the literal behavior specified by the task ("If
        # parse_react_turn returns None... set row.meta['parsed_from_text']
        # = True"), applied uniformly per-turn with no per-row aggregation.
        # Consequence for downstream use: this flag means "at least one
        # assistant turn in this row wasn't structured," NOT "this row has
        # no structured tool calls" -- don't use it alone to filter out rows
        # that are actually rich ReAct data with a normal terminal answer.
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta.get("parsed_from_text") is True


class TestAgentFlanMalformedActionInput:
    def test_lenient_fallback_wraps_raw_text(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[MALFORMED_JSON_TRAJECTORY])
        rows, stats = norm.run()
        assert stats.emitted == 1
        row = rows[0]
        call_msg = [m for m in row.messages if m.role == "assistant"][0]
        assert call_msg.tool_calls is not None
        call = call_msg.tool_calls[0]
        assert json.loads(call.function.arguments) == {"input": "{city: Paris}"}

    def test_args_raw_meta_flag_set(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[MALFORMED_JSON_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta.get("args_raw") is True


class TestAgentFlanPlainTextAssistantTurn:
    def test_row_not_dropped(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[PLAIN_TEXT_TRAJECTORY])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0

    def test_content_preserved_unchanged(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[PLAIN_TEXT_TRAJECTORY])
        rows, _ = norm.run()
        assistant_msg = [m for m in rows[0].messages if m.role == "assistant"][0]
        assert assistant_msg.content == "Sure, I can help with that. What do you need?"
        assert assistant_msg.tool_calls is None

    def test_parsed_from_text_meta_flag_set(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[PLAIN_TEXT_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta.get("parsed_from_text") is True


class TestAgentFlanOrdinaryUserTurnNotMisclassified:
    def test_non_observation_user_turn_stays_user_role(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[ORDINARY_USER_TRAJECTORY])
        rows, _ = norm.run()
        roles = [m.role for m in rows[0].messages]
        assert roles == ["user", "assistant", "user"]
        last_user_msg = rows[0].messages[-1]
        assert last_user_msg.role == "user"
        assert last_user_msg.tool_call_id is None
        assert last_user_msg.content == "Thanks, and what about Germany?"


class TestAgentFlanSourceSplitTagging:
    def test_source_split_tagged_from_injected_split(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta["source_split"] == "agent_instruct_react"

    def test_source_split_tagged_for_a_different_split(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[MALFORMED_JSON_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].meta["source_split"] == "toolbench_react_10p"


class TestAgentFlanDefaultSplits:
    def test_default_splits_are_only_the_two_react_splits(self) -> None:
        norm = AgentFlanNormalizer()
        assert norm.splits == ("agent_instruct_react", "toolbench_react_10p")
        assert norm.splits == DEFAULT_REACT_SPLITS
        assert "agent_instruct_tflan" not in norm.splits
        assert "toolbench_tflan_60p_r10r5u7" not in norm.splits
        assert "toolbench_tflan_cot_30p" not in norm.splits
        assert "toolbench_instruct_j1s1_3k" not in norm.splits
        assert "toolbench_negative" not in norm.splits

    def test_splits_override_is_respected(self) -> None:
        norm = AgentFlanNormalizer(splits=("agent_instruct_tflan",))
        assert norm.splits == ("agent_instruct_tflan",)


class TestAgentFlanSynthesizeToolSchemasNotImplemented:
    def test_raises_on_construction(self) -> None:
        # Documented choice: raised eagerly in __init__ (not deferred to
        # first use / iter_raw / to_canonical) so a misconfigured build
        # fails fast before any dataset loading happens.
        with pytest.raises(NotImplementedError):
            AgentFlanNormalizer(synthesize_tool_schemas=True)

    def test_error_message_explains_why(self) -> None:
        with pytest.raises(NotImplementedError, match="tool schema"):
            AgentFlanNormalizer(synthesize_tool_schemas=True)

    def test_default_tools_are_empty(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[CLEAN_TRAJECTORY])
        rows, _ = norm.run()
        assert rows[0].tools == []


class TestAgentFlanEmptyConversationDropped:
    def test_empty_conversation_dropped(self) -> None:
        norm = AgentFlanNormalizer(
            raw_examples=[{"id": "empty_1", "_split": "agent_instruct_react", "conversation": []}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_missing_conversation_key_dropped(self) -> None:
        norm = AgentFlanNormalizer(raw_examples=[{"id": "missing_1", "_split": "agent_instruct_react"}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestAgentFlanStatsAccounting:
    def test_mixed_batch_counts(self) -> None:
        examples = [
            CLEAN_TRAJECTORY,
            MALFORMED_JSON_TRAJECTORY,
            PLAIN_TEXT_TRAJECTORY,
            ORDINARY_USER_TRAJECTORY,
            {"id": "empty_1", "_split": "agent_instruct_react", "conversation": []},
            {"id": "missing_1", "_split": "agent_instruct_react"},
        ]
        norm = AgentFlanNormalizer(raw_examples=examples)
        rows, stats = norm.run()
        assert stats.total_raw == 6
        assert stats.emitted == 4
        assert stats.dropped == 2
        assert len(rows) == 4
