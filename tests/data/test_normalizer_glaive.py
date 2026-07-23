from agentforge.data.normalizers.glaive import GlaiveNormalizer

SYSTEM_WITH_ONE_TOOL = (
    "You are a helpful assistant with access to the following function. "
    "Use it if required -\n"
    '{"name": "get_weather", "description": "Get current weather for a city", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}'
)

HAPPY_PATH_CHAT = (
    "USER: What's the weather in Paris?\n"
    "ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": "
    '\'{"city": "Paris"}\'}\n'
    "FUNCTION RESPONSE: {\"temperature\": \"18C\", \"condition\": \"cloudy\"}\n"
    "ASSISTANT: It's 18C and cloudy in Paris."
)


class TestGlaiveHappyPath:
    def test_produces_one_row(self) -> None:
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": HAPPY_PATH_CHAT}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0
        assert len(rows) == 1

    def test_tool_extracted(self) -> None:
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": HAPPY_PATH_CHAT}])
        rows, _ = norm.run()
        row = rows[0]
        assert len(row.tools) == 1
        assert row.tools[0].function["name"] == "get_weather"

    def test_turn_order_and_roles(self) -> None:
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": HAPPY_PATH_CHAT}])
        rows, _ = norm.run()
        roles = [m.role for m in rows[0].messages]
        # A synthetic system message is prepended from the leading prose left
        # over after tool-def JSON is stripped out of `system`.
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_tool_call_linked_to_response(self) -> None:
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": HAPPY_PATH_CHAT}])
        rows, _ = norm.run()
        msgs = rows[0].messages
        assistant_call = msgs[2]
        tool_response = msgs[3]
        assert assistant_call.tool_calls is not None
        assert assistant_call.tool_calls[0].id == tool_response.tool_call_id
        assert tool_response.name == "get_weather"


class TestGlaiveMalformedJson:
    def test_malformed_functioncall_json_drops_row(self) -> None:
        bad_chat = (
            "USER: weather in Rome?\n"
            'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": {city: Rome}}'
        )
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": bad_chat}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestGlaiveNoToolCall:
    def test_plain_conversation_no_tools(self) -> None:
        chat = "USER: hello\nASSISTANT: hi there, how can I help?"
        norm = GlaiveNormalizer(raw_examples=[{"system": "You are a helpful assistant.", "chat": chat}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert rows[0].tools == []
        assert all(m.tool_calls is None for m in rows[0].messages)


class TestGlaiveMultiTurnMultipleCalls:
    def test_two_sequential_tool_calls(self) -> None:
        chat = (
            "USER: weather in Paris then Rome?\n"
            'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Paris"}\'}\n'
            'FUNCTION RESPONSE: {"temperature": "18C"}\n'
            'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Rome"}\'}\n'
            'FUNCTION RESPONSE: {"temperature": "24C"}\n'
            "ASSISTANT: Paris is 18C, Rome is 24C."
        )
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": chat}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        tool_msgs = [m for m in rows[0].messages if m.role == "tool"]
        assert len(tool_msgs) == 2
        # each tool response should link back to a distinct call id
        assert tool_msgs[0].tool_call_id != tool_msgs[1].tool_call_id


class TestGlaiveEmptyChat:
    def test_empty_chat_dropped(self) -> None:
        norm = GlaiveNormalizer(raw_examples=[{"system": SYSTEM_WITH_ONE_TOOL, "chat": ""}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestGlaiveStatsAccounting:
    def test_mixed_batch_counts(self) -> None:
        examples = [
            {"system": SYSTEM_WITH_ONE_TOOL, "chat": HAPPY_PATH_CHAT},
            {"system": SYSTEM_WITH_ONE_TOOL, "chat": ""},
            {
                "system": SYSTEM_WITH_ONE_TOOL,
                "chat": 'USER: x\nASSISTANT: <functioncall> {"name": "f", "arguments": {bad}}',
            },
        ]
        norm = GlaiveNormalizer(raw_examples=examples)
        rows, stats = norm.run()
        assert stats.total_raw == 3
        assert stats.emitted == 1
        assert stats.dropped == 2
