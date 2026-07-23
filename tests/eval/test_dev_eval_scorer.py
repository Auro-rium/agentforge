import json

from agentforge.data.schema import FunctionCall, Message, Row, ToolCall
from agentforge.eval.dev_eval import aggregate_scores, score_row


def _row_with_tool_call(name: str = "get_weather", arguments: dict | None = None) -> Row:
    arguments = arguments if arguments is not None else {"city": "Paris"}
    return Row(
        id="r1",
        source="glaive",
        messages=[
            Message(role="user", content="weather in paris?"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="call_0", function=FunctionCall(name=name, arguments=json.dumps(arguments)))
                ],
            ),
        ],
    )


def _row_with_terminal_answer(answer: str = "It's 18C and sunny in Paris.") -> Row:
    return Row(
        id="r2",
        source="agent_flan",
        messages=[
            Message(role="user", content="what's the weather?"),
            Message(role="assistant", content=answer),
        ],
    )


class TestScoreRowToolCallExpected:
    def test_exact_match(self) -> None:
        row = _row_with_tool_call()
        generated = '<|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = score_row(generated, row)
        assert result["json_valid"] is True
        assert result["name_match"] == 1.0
        assert result["arg_match_score"] == 1.0
        assert result["task_success_proxy"] is True

    def test_wrong_function_name(self) -> None:
        row = _row_with_tool_call()
        generated = '<|tool_call>{"name": "get_time", "arguments": {"city": "Paris"}}'
        result = score_row(generated, row)
        assert result["name_match"] == 0.0
        assert result["task_success_proxy"] is False

    def test_wrong_arguments(self) -> None:
        row = _row_with_tool_call()
        generated = '<|tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}'
        result = score_row(generated, row)
        assert result["name_match"] == 1.0
        assert result["arg_match_score"] == 0.0
        assert result["task_success_proxy"] is False

    def test_partial_argument_overlap(self) -> None:
        row = _row_with_tool_call(arguments={"city": "Paris", "units": "metric"})
        generated = '<|tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "units": "imperial"}}'
        result = score_row(generated, row)
        assert result["arg_match_score"] == 0.5

    def test_no_call_generated_at_all(self) -> None:
        row = _row_with_tool_call()
        generated = "Sorry, I don't know how to check the weather."
        result = score_row(generated, row)
        assert result["json_valid"] is False
        assert result["parsed_call_count"] == 0
        assert result["task_success_proxy"] is False

    def test_malformed_json_counts_as_invalid(self) -> None:
        row = _row_with_tool_call()
        generated = "<|tool_call>{not valid json at all"
        result = score_row(generated, row)
        assert result["json_valid"] is False


class TestScoreRowTerminalAnswerExpected:
    def test_matching_answer_content(self) -> None:
        row = _row_with_terminal_answer("It's 18C and sunny in Paris.")
        generated = "Based on the weather report, it's 18c and sunny in paris."
        result = score_row(generated, row)
        assert result["expected_call_count"] == 0
        assert result["task_success_proxy"] is True

    def test_non_matching_answer_content(self) -> None:
        row = _row_with_terminal_answer("It's 18C and sunny in Paris.")
        generated = "I have no idea what the weather is."
        result = score_row(generated, row)
        assert result["task_success_proxy"] is False

    def test_no_spurious_tool_call_scores_json_valid_true(self) -> None:
        row = _row_with_terminal_answer("Sure, happy to help.")
        generated = "Sure, happy to help."
        result = score_row(generated, row)
        assert result["json_valid"] is True

    def test_spurious_tool_call_scores_json_valid_false(self) -> None:
        row = _row_with_terminal_answer("Sure, happy to help.")
        generated = 'Sure, happy to help. <|tool_call>{"name": "unnecessary", "arguments": {}}'
        result = score_row(generated, row)
        assert result["json_valid"] is False


class TestAggregateScores:
    def test_empty_list(self) -> None:
        assert aggregate_scores([]) == {"count": 0}

    def test_mixed_batch_aggregation(self) -> None:
        rows_and_generations = [
            (_row_with_tool_call(), '<|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'),
            (_row_with_tool_call(), '<|tool_call>{"name": "wrong", "arguments": {}}'),
            (_row_with_terminal_answer("done"), "done"),
        ]
        scores = [score_row(gen, row) for row, gen in rows_and_generations]
        summary = aggregate_scores(scores)
        assert summary["count"] == 3
        assert summary["tool_call_rows"]["count"] == 2
        assert summary["terminal_answer_rows"]["count"] == 1
        assert 0.0 < summary["task_success_rate"] < 1.0
