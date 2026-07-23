import json

import pytest

from agentforge.data.normalizers.xlam import XlamNormalizer

TOOLS_ONE = json.dumps(
    [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
)

TOOLS_TWO = json.dumps(
    [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "convert_currency",
            "description": "Convert an amount between currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    ]
)

ANSWERS_ONE = json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])

ANSWERS_TWO = json.dumps(
    [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
        {"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}},
    ]
)


class TestXlamHappyPath:
    def test_produces_one_row(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "What's the weather in Paris?", "tools": TOOLS_ONE, "answers": ANSWERS_ONE}]
        )
        rows, stats = norm.run()
        assert stats.emitted == 1
        assert stats.dropped == 0
        assert len(rows) == 1

    def test_message_shape(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "What's the weather in Paris?", "tools": TOOLS_ONE, "answers": ANSWERS_ONE}]
        )
        rows, _ = norm.run()
        row = rows[0]
        assert row.source == "xlam"
        assert [m.role for m in row.messages] == ["user", "assistant"]
        user_msg, assistant_msg = row.messages
        assert user_msg.content == "What's the weather in Paris?"
        assert assistant_msg.content is None
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) == 1

    def test_tool_call_fields(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "What's the weather in Paris?", "tools": TOOLS_ONE, "answers": ANSWERS_ONE}]
        )
        rows, _ = norm.run()
        tool_call = rows[0].messages[1].tool_calls[0]
        assert tool_call.id == "call_0"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"city": "Paris"}

    def test_tools_extracted(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "What's the weather in Paris?", "tools": TOOLS_ONE, "answers": ANSWERS_ONE}]
        )
        rows, _ = norm.run()
        assert len(rows[0].tools) == 1
        assert rows[0].tools[0].function["name"] == "get_weather"


class TestXlamMultipleToolCalls:
    def test_two_calls_in_one_answers_list(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[
                {
                    "query": "Weather in Paris and convert 100 USD to EUR",
                    "tools": TOOLS_TWO,
                    "answers": ANSWERS_TWO,
                }
            ]
        )
        rows, stats = norm.run()
        assert stats.emitted == 1
        assistant_msg = rows[0].messages[1]
        assert len(assistant_msg.tool_calls) == 2
        assert assistant_msg.tool_calls[0].id == "call_0"
        assert assistant_msg.tool_calls[1].id == "call_1"
        assert assistant_msg.tool_calls[0].function.name == "get_weather"
        assert assistant_msg.tool_calls[1].function.name == "convert_currency"
        assert len(rows[0].tools) == 2


class TestXlamMalformedJson:
    def test_malformed_answers_json_drops_row(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": "{not valid json"}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_malformed_tools_json_drops_row(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[{"query": "weather in Rome?", "tools": "[{bad json}]", "answers": ANSWERS_ONE}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_answers_not_a_list_drops_row(self) -> None:
        norm = XlamNormalizer(
            raw_examples=[
                {"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": json.dumps({"name": "get_weather"})}
            ]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_tool_missing_name_drops_row(self) -> None:
        bad_tools = json.dumps([{"description": "no name field", "parameters": {}}])
        norm = XlamNormalizer(
            raw_examples=[{"query": "weather in Rome?", "tools": bad_tools, "answers": ANSWERS_ONE}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_answer_missing_name_drops_row(self) -> None:
        bad_answers = json.dumps([{"arguments": {"city": "Rome"}}])
        norm = XlamNormalizer(
            raw_examples=[{"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": bad_answers}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestXlamEmptyAnswers:
    def test_empty_answers_list_drops_row(self) -> None:
        # xLAM's `answers` is (almost) never empty in practice, but if it were,
        # an assistant message with content=None and tool_calls=[] carries no
        # learnable signal -- nothing for the model to imitate -- so we drop
        # the row rather than emit a degenerate turn. This is a deliberate,
        # documented choice (see xlam.py:to_canonical), not an oversight.
        norm = XlamNormalizer(
            raw_examples=[{"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": json.dumps([])}]
        )
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert stats.drop_reasons["to_canonical_returned_none"] == 1
        assert len(rows) == 0


class TestXlamMissingFields:
    def test_empty_query_dropped(self) -> None:
        norm = XlamNormalizer(raw_examples=[{"query": "  ", "tools": TOOLS_ONE, "answers": ANSWERS_ONE}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0

    def test_missing_tools_field_dropped(self) -> None:
        norm = XlamNormalizer(raw_examples=[{"query": "weather?", "answers": ANSWERS_ONE}])
        rows, stats = norm.run()
        assert stats.dropped == 1
        assert len(rows) == 0


class TestXlamStatsAccounting:
    def test_mixed_batch_counts(self) -> None:
        examples = [
            {"query": "What's the weather in Paris?", "tools": TOOLS_ONE, "answers": ANSWERS_ONE},
            {
                "query": "Weather in Paris and convert 100 USD to EUR",
                "tools": TOOLS_TWO,
                "answers": ANSWERS_TWO,
            },
            {"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": "{not valid json"},
            {"query": "weather in Rome?", "tools": TOOLS_ONE, "answers": json.dumps([])},
            {"query": "", "tools": TOOLS_ONE, "answers": ANSWERS_ONE},
        ]
        norm = XlamNormalizer(raw_examples=examples)
        rows, stats = norm.run()
        assert stats.total_raw == 5
        assert stats.emitted == 2
        assert stats.dropped == 3
        assert len(rows) == 2


class TestXlamGatedDatasetErrorWrapping:
    def test_real_gated_repo_error_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from huggingface_hub.errors import GatedRepoError
        from unittest.mock import Mock

        fake_response = Mock()
        fake_response.status_code = 403

        class _FakeDatasetsModule:
            @staticmethod
            def load_dataset(*args, **kwargs):
                raise GatedRepoError(
                    "Access to dataset Salesforce/xlam-function-calling-60k is restricted. "
                    "You must have access to it and be authenticated to access it.",
                    response=fake_response,
                )

        import sys

        monkeypatch.setitem(sys.modules, "datasets", _FakeDatasetsModule())

        norm = XlamNormalizer()
        with pytest.raises(RuntimeError) as exc_info:
            list(norm.iter_raw())

        message = str(exc_info.value)
        assert "gated" in message.lower()
        assert "HF_TOKEN" in message
        assert "huggingface.co/datasets/Salesforce/xlam-function-calling-60k" in message
        # original exception is preserved for debugging
        assert isinstance(exc_info.value.__cause__, GatedRepoError)

    def test_generic_401_error_is_also_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Fallback path: some other exception type (not GatedRepoError) but
        # with wording that clearly indicates a gated/auth failure.
        class _FakeDatasetsModule:
            @staticmethod
            def load_dataset(*args, **kwargs):
                raise OSError(
                    "401 Client Error: Unauthorized for url: "
                    "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"
                )

        import sys

        monkeypatch.setitem(sys.modules, "datasets", _FakeDatasetsModule())

        norm = XlamNormalizer()
        with pytest.raises(RuntimeError) as exc_info:
            list(norm.iter_raw())

        message = str(exc_info.value)
        assert "gated" in message.lower()
        assert "HF_TOKEN" in message

    def test_unrelated_error_is_not_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A totally unrelated failure (e.g. no network) should propagate as-is,
        # not get relabeled as a gating problem.
        class _FakeDatasetsModule:
            @staticmethod
            def load_dataset(*args, **kwargs):
                raise ConnectionError("could not resolve host")

        import sys

        monkeypatch.setitem(sys.modules, "datasets", _FakeDatasetsModule())

        norm = XlamNormalizer()
        with pytest.raises(ConnectionError):
            list(norm.iter_raw())
