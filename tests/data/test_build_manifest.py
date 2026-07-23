import json
from pathlib import Path

import pytest

from agentforge.config import AgentForgeConfig, DataConfig, ModelConfig, TrainingConfig
from agentforge.data.build_manifest import process_rows
from agentforge.data.normalizers.base import NormalizationStats
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall


def _plain_row(source: str, i: int) -> Row:
    return Row(
        id=f"{source}_{i}", source=source, messages=[Message(role="user", content=f"{source} msg {i}")]
    )


def _tool_call_row(source: str, i: int) -> Row:
    return Row(
        id=f"{source}_tc_{i}",
        source=source,
        messages=[
            Message(role="user", content=f"{source} weather? {i}"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_0", function=FunctionCall(name="f", arguments="{}"))],
            ),
        ],
    )


def _make_cfg(tmp_path: Path, sources: dict[str, float]) -> AgentForgeConfig:
    return AgentForgeConfig(
        run_name="test",
        model=ModelConfig(base_model="test/model"),
        training=TrainingConfig(output_dir=str(tmp_path / "outputs")),
        data=DataConfig(
            manifest_path=str(tmp_path / "manifest.jsonl"),
            holdout_path=str(tmp_path / "holdout.jsonl"),
            sources=sources,
            shuffle=True,
        ),
        seed=0,
    )


def _empty_stats(source: str, n: int) -> NormalizationStats:
    return NormalizationStats(source=source, total_raw=n, emitted=n, dropped=0)


class TestProcessRowsBasicPipeline:
    def test_writes_manifest_and_holdout_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        raw = {"glaive": [_plain_row("glaive", i) for i in range(20)]}
        stats_in = {"glaive": _empty_stats("glaive", 20)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0})
        process_rows(cfg, raw, stats_in, holdout_size=5)

        manifest_path = Path(cfg.data.manifest_path)
        holdout_path = Path(cfg.data.holdout_path)
        assert manifest_path.exists()
        assert holdout_path.exists()

    def test_train_and_holdout_disjoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        raw = {"glaive": [_plain_row("glaive", i) for i in range(20)]}
        stats_in = {"glaive": _empty_stats("glaive", 20)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0})
        process_rows(cfg, raw, stats_in, holdout_size=5)

        manifest_ids = {
            json.loads(line)["id"] for line in Path(cfg.data.manifest_path).read_text().splitlines()
        }
        holdout_ids = {
            json.loads(line)["id"] for line in Path(cfg.data.holdout_path).read_text().splitlines()
        }
        assert manifest_ids.isdisjoint(holdout_ids)

    def test_weighting_applied_in_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        raw = {
            "glaive": [_plain_row("glaive", i) for i in range(20)],
            "agent_flan": [_plain_row("agent_flan", i) for i in range(10)],
        }
        stats_in = {"glaive": _empty_stats("glaive", 20), "agent_flan": _empty_stats("agent_flan", 10)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0, "agent_flan": 3.0})
        stats = process_rows(cfg, raw, stats_in, holdout_size=0)

        # agent_flan should end up ~3x its (post-holdout) pool size in the manifest
        assert stats["sources"]["agent_flan"]["post_weight_in_manifest"] == 30
        assert stats["sources"]["glaive"]["post_weight_in_manifest"] == 20

    def test_exact_duplicate_rows_removed_across_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        dup = Message(role="user", content="identical content")
        row_a = Row(id="a", source="glaive", messages=[dup])
        row_b = Row(id="b", source="hermes", messages=[dup])  # same messages, different id/source
        raw = {"glaive": [row_a], "hermes": [row_b]}
        stats_in = {"glaive": _empty_stats("glaive", 1), "hermes": _empty_stats("hermes", 1)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0, "hermes": 1.0})
        stats = process_rows(cfg, raw, stats_in, holdout_size=0)

        assert stats["dedup_removed_total"] == 1
        assert stats["manifest_total_rows"] == 1

    def test_manifest_stats_json_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        raw = {"glaive": [_plain_row("glaive", i) for i in range(10)]}
        stats_in = {"glaive": _empty_stats("glaive", 10)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0})
        process_rows(cfg, raw, stats_in, holdout_size=2)

        stats_path = tmp_path / "data" / "manifest_stats.json"
        assert stats_path.exists()
        loaded = json.loads(stats_path.read_text())
        assert loaded["sources"]["glaive"]["raw"] == 10

    def test_row_level_metric_pct_with_tool_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        raw = {
            "glaive": [_plain_row("glaive", i) for i in range(5)] + [_tool_call_row("glaive", i) for i in range(5)]
        }
        stats_in = {"glaive": _empty_stats("glaive", 10)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0})
        stats = process_rows(cfg, raw, stats_in, holdout_size=0)

        assert stats["manifest"]["pct_with_tool_calls"] == 50.0

    def test_max_examples_per_source_not_applied_here(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # max_examples_per_source is applied in run_normalizers (the
        # network-touching step), not process_rows -- confirm process_rows
        # doesn't re-truncate on its own.
        monkeypatch.chdir(tmp_path)
        raw = {"glaive": [_plain_row("glaive", i) for i in range(10)]}
        stats_in = {"glaive": _empty_stats("glaive", 10)}
        cfg = _make_cfg(tmp_path, {"glaive": 1.0})
        cfg.data.max_examples_per_source = 3
        stats = process_rows(cfg, raw, stats_in, holdout_size=0)
        assert stats["manifest_total_rows"] == 10
