import json

from agentforge.data.normalizers.hermes import HermesNormalizer

SYSTEM_WITH_TOOLS = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. Call one or more functions "
    "if needed.\n"
    "<tools>"
    '[{"name": "get_weather", "description": "Get current weather for a city", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
    '"required": ["city"]}}]'
    "</tools>\n"
    "Use the following pydantic model json schema for each tool call you make."
)

HAPPY_PATH_CONVERSATIONS = [
    {"from": "system", "value": SYSTEM_WITH_TOOLS},
    {"from": "human", "value": "What's the weather in Paris?"},
    {
        "from": "gpt",
        "value": (
            "<tool_call>"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}'
            "</tool_call>"
        ),
    },
    {
        "from": "tool",
        "value": (
            "<tool_response>"
            '{"name": "get_weather", "content": {"temperature": "18C", "condition": "cloudy"}}'
            "</tool_response>"
        ),
    },
    {"from": "gpt", "value": "It's 18C and cloudy in Paris."},
]


def _happy_path_raw(config: str = "func-calling") -> dict:
    return {"conversations": HAPPY_PATH_CONVERSATIONS, "_hermes_config": config}


class TestHermesHappyPath:
    def test_produces_one_row(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0
        assert len(rows) == 1

    def test_tool_extracted(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, _ = norm.run()
        row = rows[0]
        assert len(row.tools) == 1
        assert row.tools[0].function["name"] == "get_weather"

    def test_turn_order_and_roles(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, _ = norm.run()
        roles = [m.role for m in rows[0].messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_system_prose_kept_and_populated_tag_stripped(self) -> None:
        # The fixture's system prompt mentions the tag name descriptively
        # ("within <tools></tools> XML tags") *and* has the real, populated
        # tag later -- only the populated one (with the tool-def JSON) should
        # be stripped; the descriptive mention is just prose and stays.
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, _ = norm.run()
        system_msg = rows[0].messages[0]
        assert '"get_weather"' not in system_msg.content
        assert "function calling AI model" in system_msg.content

    def test_tool_call_parsed_and_linked_to_response(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, _ = norm.run()
        msgs = rows[0].messages
        assistant_call = msgs[2]
        tool_response = msgs[3]
        assert assistant_call.tool_calls is not None
        assert len(assistant_call.tool_calls) == 1
        call = assistant_call.tool_calls[0]
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "Paris"}
        assert call.id == tool_response.tool_call_id
        assert tool_response.name == "get_weather"
        assert json.loads(tool_response.content) == {"temperature": "18C", "condition": "cloudy"}

    def test_tool_call_tag_stripped_from_assistant_content(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw()])
        rows, _ = norm.run()
        assistant_call = rows[0].messages[2]
        assert assistant_call.content is None or "<tool_call>" not in assistant_call.content


class TestHermesToolResponseEmbeddedInHumanTurn:
    def test_human_turn_with_tool_response_tag_remapped_to_tool_role(self) -> None:
        conversations = [
            {"from": "system", "value": SYSTEM_WITH_TOOLS},
            {"from": "human", "value": "What's the weather in Rome?"},
            {
                "from": "gpt",
                "value": (
                    "<tool_call>"
                    '{"name": "get_weather", "arguments": {"city": "Rome"}}'
                    "</tool_call>"
                ),
            },
            {
                # Embedded in a "human" turn, not a separate "tool" role.
                "from": "human",
                "value": (
                    "<tool_response>"
                    '{"name": "get_weather", "content": {"temperature": "24C"}}'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "It's 24C in Rome."},
        ]
        norm = HermesNormalizer(
            raw_examples=[{"conversations": conversations, "_hermes_config": "func-calling"}]
        )
        rows, stats = norm.run()
        assert stats.emitted == 1
        roles = [m.role for m in rows[0].messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        tool_msg = rows[0].messages[3]
        assert tool_msg.name == "get_weather"
        assert json.loads(tool_msg.content) == {"temperature": "24C"}
        assert tool_msg.tool_call_id == rows[0].messages[2].tool_calls[0].id


class TestHermesNoToolsPlainConversation:
    def test_plain_conversation_no_tools_still_valid(self) -> None:
        conversations = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi there, how can I help?"},
        ]
        norm = HermesNormalizer(
            raw_examples=[{"conversations": conversations, "_hermes_config": "func-calling-singleturn"}]
        )
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert rows[0].tools == []
        assert all(m.tool_calls is None for m in rows[0].messages)
        assert [m.role for m in rows[0].messages] == ["system", "user", "assistant"]


class TestHermesMalformedToolCall:
    def test_unparseable_tool_call_json_drops_row(self) -> None:
        conversations = [
            {"from": "system", "value": SYSTEM_WITH_TOOLS},
            {"from": "human", "value": "weather in Berlin?"},
            {
                "from": "gpt",
                "value": '<tool_call>{"name": "get_weather", "arguments": {city: Berlin}}</tool_call>',
            },
        ]
        norm = HermesNormalizer(
            raw_examples=[{"conversations": conversations, "_hermes_config": "func-calling"}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_tool_call_missing_name_drops_row(self) -> None:
        conversations = [
            {"from": "system", "value": SYSTEM_WITH_TOOLS},
            {"from": "human", "value": "weather in Berlin?"},
            {
                "from": "gpt",
                "value": '<tool_call>{"arguments": {"city": "Berlin"}}</tool_call>',
            },
        ]
        norm = HermesNormalizer(
            raw_examples=[{"conversations": conversations, "_hermes_config": "func-calling"}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestHermesConfigTagging:
    def test_hermes_config_tagged_in_meta(self) -> None:
        norm = HermesNormalizer(raw_examples=[_happy_path_raw(config="json-mode-agentic")])
        rows, _ = norm.run()
        assert rows[0].meta["hermes_config"] == "json-mode-agentic"

    def test_missing_config_key_defaults_to_unknown(self) -> None:
        norm = HermesNormalizer(raw_examples=[{"conversations": HAPPY_PATH_CONVERSATIONS}])
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert rows[0].meta["hermes_config"] == "unknown"


class TestHermesStatsAccounting:
    def test_mixed_batch_counts(self) -> None:
        bad_json_conversations = [
            {"from": "system", "value": SYSTEM_WITH_TOOLS},
            {"from": "human", "value": "x"},
            {"from": "gpt", "value": '<tool_call>{"name": "f", "arguments": {bad}}</tool_call>'},
        ]
        examples = [
            _happy_path_raw(),
            {"conversations": [], "_hermes_config": "func-calling"},  # empty -> drop
            {"conversations": bad_json_conversations, "_hermes_config": "func-calling"},  # malformed -> drop
            {
                "conversations": [
                    {"from": "system", "value": "You are a helpful assistant."},
                    {"from": "human", "value": "hi"},
                    {"from": "gpt", "value": "hello!"},
                ],
                "_hermes_config": "func-calling-singleturn",
            },  # plain, no tools -> emit
        ]
        norm = HermesNormalizer(raw_examples=examples)
        rows, stats = norm.run()
        assert stats.total_raw == 4
        assert stats.emitted == 2
        assert stats.dropped == 2
