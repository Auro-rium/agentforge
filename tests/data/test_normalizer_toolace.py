from agentforge.data.normalizers.toolace import ToolACENormalizer

SYSTEM_WITH_ONE_TOOL = (
    "You are a helpful assistant with access to the following tools. "
    "Use them if required -\n"
    '{"name": "get_weather", "description": "Get current weather for a city", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}'
)

SYSTEM_WITH_TWO_TOOLS = (
    "You are a helpful assistant with access to the following tools.\n"
    '{"name": "get_weather", "description": "Get current weather for a city", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}\n'
    '{"name": "get_time", "description": "Get current time for a city", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}'
)

HAPPY_PATH_CONVERSATIONS = [
    {"from": "human", "value": "What's the weather in Paris?"},
    {"from": "gpt", "value": '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'},
    {"from": "human", "value": '{"temperature": "18C", "condition": "cloudy"}'},
    {"from": "gpt", "value": "It's 18C and cloudy in Paris."},
]


class TestToolACEHappyPath:
    def test_produces_one_row(self) -> None:
        norm = ToolACENormalizer(
            raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS}]
        )
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0
        assert len(rows) == 1

    def test_tool_extracted(self) -> None:
        norm = ToolACENormalizer(
            raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS}]
        )
        rows, _ = norm.run()
        row = rows[0]
        assert len(row.tools) == 1
        assert row.tools[0].function["name"] == "get_weather"

    def test_turn_order_and_roles(self) -> None:
        norm = ToolACENormalizer(
            raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS}]
        )
        rows, _ = norm.run()
        roles = [m.role for m in rows[0].messages]
        # A synthetic system message is prepended from the leading prose left
        # over after tool-def JSON is stripped out of `system`. The `human`
        # turn right after the tool-calling `gpt` turn is remapped to `tool`.
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_gpt_turn_parsed_into_tool_calls(self) -> None:
        norm = ToolACENormalizer(
            raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS}]
        )
        rows, _ = norm.run()
        assistant_call = rows[0].messages[2]
        assert assistant_call.role == "assistant"
        assert assistant_call.content is None
        assert assistant_call.tool_calls is not None
        assert len(assistant_call.tool_calls) == 1
        assert assistant_call.tool_calls[0].function.name == "get_weather"

    def test_human_after_tool_call_remapped_and_linked(self) -> None:
        norm = ToolACENormalizer(
            raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS}]
        )
        rows, _ = norm.run()
        assistant_call = rows[0].messages[2]
        tool_response = rows[0].messages[3]
        assert tool_response.role == "tool"
        assert tool_response.tool_call_id == assistant_call.tool_calls[0].id
        assert tool_response.name == "get_weather"
        assert tool_response.content == '{"temperature": "18C", "condition": "cloudy"}'


class TestToolACEPlainProse:
    def test_plain_prose_gpt_turn_has_no_tool_calls(self) -> None:
        conversations = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi there, how can I help?"},
        ]
        norm = ToolACENormalizer(raw_examples=[{"system": "You are a helpful assistant.", "conversations": conversations}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert rows[0].tools == []
        assert all(m.tool_calls is None for m in rows[0].messages)
        gpt_msg = rows[0].messages[-1]
        assert gpt_msg.role == "assistant"
        assert gpt_msg.content == "hi there, how can I help?"

    def test_human_after_plain_prose_assistant_stays_user(self) -> None:
        # Only a `human` turn right after a tool-calling assistant turn
        # remaps; a `human` turn after ordinary prose must stay `user`.
        conversations = [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi there, how can I help?"},
            {"from": "human", "value": "what's the weather in Rome?"},
        ]
        norm = ToolACENormalizer(raw_examples=[{"system": "You are a helpful assistant.", "conversations": conversations}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        roles = [m.role for m in rows[0].messages]
        # Leading "system" is the synthetic system message carried over from
        # the (non-JSON, all-prose) `system` column; the point under test is
        # that the two `human` turns both stay `user`, since neither follows
        # a tool-calling assistant turn.
        assert roles == ["system", "user", "assistant", "user"]
        assert rows[0].messages[-1].content == "what's the weather in Rome?"


class TestToolACEToolDefExtraction:
    def test_multiple_tool_defs_extracted(self) -> None:
        conversations = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello!"},
        ]
        norm = ToolACENormalizer(raw_examples=[{"system": SYSTEM_WITH_TWO_TOOLS, "conversations": conversations}])
        rows, _ = norm.run()
        names = {t.function["name"] for t in rows[0].tools}
        assert names == {"get_weather", "get_time"}


class TestToolACEMultiToolCallTurn:
    def test_two_calls_in_one_gpt_turn(self) -> None:
        conversations = [
            {"from": "human", "value": "weather and time in Paris?"},
            {
                "from": "gpt",
                "value": (
                    '[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
                    '{"name": "get_time", "arguments": {"city": "Paris"}}]'
                ),
            },
        ]
        norm = ToolACENormalizer(raw_examples=[{"system": SYSTEM_WITH_TWO_TOOLS, "conversations": conversations}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assistant_call = rows[0].messages[-1]
        assert assistant_call.role == "assistant"
        assert len(assistant_call.tool_calls) == 2
        call_names = [tc.function.name for tc in assistant_call.tool_calls]
        assert call_names == ["get_weather", "get_time"]
        # each call gets a distinct synthetic id
        ids = {tc.id for tc in assistant_call.tool_calls}
        assert len(ids) == 2


class TestToolACEEmptyConversation:
    def test_empty_conversations_dropped(self) -> None:
        norm = ToolACENormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "conversations": []}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestToolACEStatsAccounting:
    def test_mixed_batch_counts(self) -> None:
        examples = [
            {"system": SYSTEM_WITH_ONE_TOOL, "conversations": HAPPY_PATH_CONVERSATIONS},
            {"system": SYSTEM_WITH_ONE_TOOL, "conversations": []},
            {
                "system": "You are a helpful assistant.",
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hi there!"},
                ],
            },
        ]
        norm = ToolACENormalizer(raw_examples=examples)
        rows, stats = norm.run()
        assert stats.total_raw == 3
        assert stats.emitted == 2
        assert stats.dropped == 1
