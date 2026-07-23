from agentforge.data.schema import Message, Row
from agentforge.eval.holdout_manifest import stratified_holdout_split


def _rows(n: int, source: str) -> list[Row]:
    return [
        Row(id=f"{source}_{i}", source=source, messages=[Message(role="user", content=f"msg {i}")])
        for i in range(n)
    ]


class TestStratifiedHoldoutSplit:
    def test_holdout_and_train_are_disjoint(self) -> None:
        rows_by_source = {"glaive": _rows(100, "glaive"), "agent_flan": _rows(50, "agent_flan")}
        remaining, holdout = stratified_holdout_split(rows_by_source, holdout_size=30, seed=0)
        train_ids = {r.id for rows in remaining.values() for r in rows}
        holdout_ids = {r.id for rows in holdout.values() for r in rows}
        assert train_ids.isdisjoint(holdout_ids)

    def test_train_plus_holdout_equals_input(self) -> None:
        rows_by_source = {"glaive": _rows(100, "glaive"), "agent_flan": _rows(50, "agent_flan")}
        remaining, holdout = stratified_holdout_split(rows_by_source, holdout_size=30, seed=0)
        for source in rows_by_source:
            assert len(remaining[source]) + len(holdout[source]) == len(rows_by_source[source])

    def test_approximately_proportional_by_source(self) -> None:
        # glaive is 4x agent_flan's size -> should get ~4x the holdout rows
        rows_by_source = {"glaive": _rows(80, "glaive"), "agent_flan": _rows(20, "agent_flan")}
        _, holdout = stratified_holdout_split(rows_by_source, holdout_size=20, seed=0)
        assert len(holdout["glaive"]) > len(holdout["agent_flan"])

    def test_every_nonempty_source_gets_at_least_one_holdout_row(self) -> None:
        rows_by_source = {"glaive": _rows(1000, "glaive"), "xlam": _rows(2, "xlam")}
        _, holdout = stratified_holdout_split(rows_by_source, holdout_size=10, seed=0)
        assert len(holdout["xlam"]) >= 1

    def test_empty_source_stays_empty(self) -> None:
        rows_by_source = {"glaive": _rows(10, "glaive"), "hermes": []}
        remaining, holdout = stratified_holdout_split(rows_by_source, holdout_size=5, seed=0)
        assert remaining["hermes"] == []
        assert holdout["hermes"] == []

    def test_zero_holdout_size_returns_all_in_train(self) -> None:
        rows_by_source = {"glaive": _rows(10, "glaive")}
        remaining, holdout = stratified_holdout_split(rows_by_source, holdout_size=0, seed=0)
        assert len(remaining["glaive"]) == 10
        assert holdout["glaive"] == []

    def test_deterministic_given_same_seed(self) -> None:
        rows_by_source = {"glaive": _rows(50, "glaive")}
        _, holdout_a = stratified_holdout_split(rows_by_source, holdout_size=10, seed=7)
        _, holdout_b = stratified_holdout_split(rows_by_source, holdout_size=10, seed=7)
        assert [r.id for r in holdout_a["glaive"]] == [r.id for r in holdout_b["glaive"]]
