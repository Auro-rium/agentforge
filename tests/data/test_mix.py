import random

from agentforge.data.mix import apply_source_weight, apply_weights_by_source
from agentforge.data.schema import Message, Row


def _rows(n: int, source: str = "glaive") -> list[Row]:
    return [
        Row(id=f"{source}_{i}", source=source, messages=[Message(role="user", content=f"msg {i}")])
        for i in range(n)
    ]


class TestApplySourceWeight:
    def test_weight_one_keeps_all_rows_once(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, 1.0, rng=random.Random(0))
        assert len(result) == 10
        assert {r.id for r in result} == {r.id for r in rows}

    def test_weight_zero_returns_empty(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, 0.0, rng=random.Random(0))
        assert result == []

    def test_negative_weight_returns_empty(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, -1.0, rng=random.Random(0))
        assert result == []

    def test_fractional_weight_undersamples_without_replacement(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, 0.5, rng=random.Random(0))
        assert len(result) == 5
        # no duplicates when undersampling
        assert len({r.id for r in result}) == 5

    def test_oversample_weight_hits_target_count(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, 3.0, rng=random.Random(0))
        assert len(result) == 30

    def test_oversample_includes_every_row_at_least_once(self) -> None:
        rows = _rows(10)
        result = apply_source_weight(rows, 2.5, rng=random.Random(0))
        assert len(result) == 25
        assert {r.id for r in rows}.issubset({r.id for r in result})

    def test_empty_input_returns_empty(self) -> None:
        assert apply_source_weight([], 3.0, rng=random.Random(0)) == []


class TestApplyWeightsBySource:
    def test_per_source_weights_applied(self) -> None:
        rows_by_source = {"glaive": _rows(10, "glaive"), "agent_flan": _rows(4, "agent_flan")}
        weights = {"glaive": 1.0, "agent_flan": 3.0}
        result = apply_weights_by_source(rows_by_source, weights, seed=0)
        assert len(result["glaive"]) == 10
        assert len(result["agent_flan"]) == 12

    def test_source_missing_from_weights_excluded(self) -> None:
        rows_by_source = {"glaive": _rows(10, "glaive"), "xlam": _rows(5, "xlam")}
        weights = {"glaive": 1.0}  # xlam intentionally omitted
        result = apply_weights_by_source(rows_by_source, weights, seed=0)
        assert result["xlam"] == []
        assert len(result["glaive"]) == 10

    def test_deterministic_given_same_seed(self) -> None:
        rows_by_source = {"glaive": _rows(10, "glaive")}
        weights = {"glaive": 0.5}
        result_a = apply_weights_by_source(rows_by_source, weights, seed=42)
        result_b = apply_weights_by_source(rows_by_source, weights, seed=42)
        assert [r.id for r in result_a["glaive"]] == [r.id for r in result_b["glaive"]]
