import pytest
from pydantic import ValidationError

from agentforge.data.schema import FunctionCall, Message, Row, ToolCall, ToolSpec


def _tool_spec(name: str = "get_weather") -> ToolSpec:
    return ToolSpec(
        function={
            "name": name,
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    )


def _tool_call(call_id: str = "call_0", name: str = "get_weather") -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments='{"city": "Paris"}'))


class TestFunctionCall:
    def test_valid_json_arguments(self) -> None:
        fc = FunctionCall(name="foo", arguments='{"x": 1}')
        assert fc.arguments == '{"x": 1}'

    def test_invalid_json_arguments_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FunctionCall(name="foo", arguments="not json")


class TestMessage:
    def test_plain_user_message(self) -> None:
        msg = Message(role="user", content="hi")
        assert msg.content == "hi"

    def test_assistant_with_tool_calls(self) -> None:
        msg = Message(role="assistant", tool_calls=[_tool_call()])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_calls_on_non_assistant_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", tool_calls=[_tool_call()])

    def test_tool_role_requires_tool_call_id(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="tool", content="42 degrees")

    def test_tool_role_valid(self) -> None:
        msg = Message(role="tool", content="42 degrees", tool_call_id="call_0", name="get_weather")
        assert msg.tool_call_id == "call_0"

    def test_name_on_non_tool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="user", content="hi", name="get_weather")


class TestToolSpec:
    def test_valid_tool_spec(self) -> None:
        spec = _tool_spec()
        assert spec.function["name"] == "get_weather"

    def test_missing_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSpec(function={"description": "no name here"})


class TestRow:
    def test_minimal_valid_row(self) -> None:
        row = Row(
            id="abc123",
            source="glaive",
            messages=[Message(role="user", content="hi")],
        )
        assert row.source == "glaive"

    def test_row_with_tool_call_and_response(self) -> None:
        row = Row(
            id="abc123",
            source="xlam",
            messages=[
                Message(role="user", content="weather in paris?"),
                Message(role="assistant", tool_calls=[_tool_call()]),
                Message(role="tool", content="42F", tool_call_id="call_0", name="get_weather"),
            ],
            tools=[_tool_spec()],
        )
        assert len(row.messages) == 3
        assert row.meta.get("orphan_tool_response") is None

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Row(id="x", source="glaive", messages=[])

    def test_first_message_must_be_system_or_user(self) -> None:
        with pytest.raises(ValidationError):
            Row(
                id="x",
                source="glaive",
                messages=[Message(role="assistant", content="hi")],
            )

    def test_orphan_tool_response_tagged_not_dropped(self) -> None:
        row = Row(
            id="x",
            source="hermes",
            messages=[
                Message(role="user", content="hi"),
                Message(role="tool", content="oops", tool_call_id="call_never_opened", name="f"),
            ],
        )
        assert row.meta["orphan_tool_response"] is True

    def test_system_first_message_allowed(self) -> None:
        row = Row(
            id="x",
            source="toolace",
            messages=[
                Message(role="system", content="you are a helpful agent"),
                Message(role="user", content="hi"),
            ],
        )
        assert row.messages[0].role == "system"
