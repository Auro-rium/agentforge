from agentforge.eval.tool_call_parsers import (
    parse_gemma4_tool_calls,
    parse_generic_tool_calls,
    parse_tool_calls,
)


class TestParseGemma4ToolCalls:
    def test_single_call(self) -> None:
        text = '<|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        calls = parse_gemma4_tool_calls(text)
        assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]

    def test_no_marker_returns_empty(self) -> None:
        assert parse_gemma4_tool_calls("just plain text, no tool calls here") == []

    def test_two_sequential_calls(self) -> None:
        text = (
            '<|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
            " some separator text "
            '<|tool_call>{"name": "get_time", "arguments": {"city": "Paris"}}'
        )
        calls = parse_gemma4_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "get_weather"
        assert calls[1]["name"] == "get_time"

    def test_nested_braces_in_arguments_handled(self) -> None:
        text = '<|tool_call>{"name": "search", "arguments": {"filter": {"nested": {"deep": 1}}}}'
        calls = parse_gemma4_tool_calls(text)
        assert calls == [{"name": "search", "arguments": {"filter": {"nested": {"deep": 1}}}}]

    def test_marker_with_no_valid_json_after_yields_nothing_for_that_marker(self) -> None:
        text = "<|tool_call>not json at all"
        assert parse_gemma4_tool_calls(text) == []

    def test_marker_with_no_json_then_marker_with_json(self) -> None:
        text = '<|tool_call>oops <|tool_call>{"name": "f", "arguments": {}}'
        calls = parse_gemma4_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "f"


class TestParseGenericToolCalls:
    def test_qwen_style_tags(self) -> None:
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>'
        calls = parse_generic_tool_calls(text)
        assert calls == [{"name": "get_weather", "arguments": {"city": "Rome"}}]

    def test_json_fenced_block(self) -> None:
        text = '```json\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n```'
        calls = parse_generic_tool_calls(text)
        assert calls == [{"name": "get_weather", "arguments": {"city": "Rome"}}]

    def test_bare_json_object_no_wrapper(self) -> None:
        text = 'Sure, calling: {"name": "get_weather", "arguments": {"city": "Rome"}}'
        calls = parse_generic_tool_calls(text)
        assert calls == [{"name": "get_weather", "arguments": {"city": "Rome"}}]

    def test_no_json_anywhere_returns_empty(self) -> None:
        assert parse_generic_tool_calls("no tool calls in this text") == []

    def test_qwen_tags_preferred_over_json_fence_when_both_present(self) -> None:
        text = (
            '<tool_call>{"name": "real_call", "arguments": {}}</tool_call>\n'
            'and also a fenced example: ```json\n{"name": "example_only", "arguments": {}}\n```'
        )
        calls = parse_generic_tool_calls(text)
        assert calls == [{"name": "real_call", "arguments": {}}]


class TestParseToolCallsDispatch:
    def test_gemma4_family_uses_gemma_parser(self) -> None:
        text = '<|tool_call>{"name": "f", "arguments": {}}'
        calls = parse_tool_calls(text, model_family="gemma4")
        assert calls == [{"name": "f", "arguments": {}}]

    def test_gemma4_family_falls_back_to_generic_when_no_gemma_markers_found(self) -> None:
        text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        calls = parse_tool_calls(text, model_family="gemma4")
        assert calls == [{"name": "f", "arguments": {}}]

    def test_other_family_uses_generic_parser_directly(self) -> None:
        text = '```json\n{"name": "f", "arguments": {}}\n```'
        calls = parse_tool_calls(text, model_family="qwen")
        assert calls == [{"name": "f", "arguments": {}}]
