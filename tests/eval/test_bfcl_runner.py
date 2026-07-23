import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from agentforge.eval.bfcl_runner import (
    collect_bfcl_scores,
    run_bfcl_eval,
    run_bfcl_evaluate,
    run_bfcl_generate,
)


class TestRunBfclGenerateCommandShape:
    def test_invokes_expected_cli_flags(self) -> None:
        with patch("agentforge.eval.bfcl_runner.subprocess.run") as mock_run:
            run_bfcl_generate(
                handler_key="gemma-4-12b-it",
                local_model_path="/models/gemma4",
                adapter_dir="/adapters/run1",
                max_lora_rank=16,
                test_categories=["multi_turn_base", "multi_turn_miss_func"],
                backend="vllm",
            )
        args = mock_run.call_args[0][0]
        assert args[:3] == ["bfcl", "generate", "--model"]
        assert "gemma-4-12b-it" in args
        assert "--backend" in args and "vllm" in args
        assert "--local-model-path" in args and "/models/gemma4" in args
        assert "--enable-lora" in args
        assert "--max-lora-rank" in args and "16" in args
        assert "--lora-modules" in args
        assert "agentforge=/adapters/run1" in args
        assert "--test-category" in args and "multi_turn_base,multi_turn_miss_func" in args

    def test_check_true_propagates_subprocess_errors(self) -> None:
        with patch("agentforge.eval.bfcl_runner.subprocess.run") as mock_run:
            run_bfcl_generate(
                handler_key="m", local_model_path="p", adapter_dir="a", max_lora_rank=8, test_categories=["simple"]
            )
        assert mock_run.call_args.kwargs.get("check") is True


class TestRunBfclEvaluateCommandShape:
    def test_invokes_expected_cli_flags(self) -> None:
        with patch("agentforge.eval.bfcl_runner.subprocess.run") as mock_run:
            run_bfcl_evaluate(handler_key="gemma-4-12b-it", test_categories=["simple", "multiple"])
        args = mock_run.call_args[0][0]
        assert args == ["bfcl", "evaluate", "--model", "gemma-4-12b-it", "--test-category", "simple,multiple"]


class TestCollectBfclScores:
    def test_missing_score_dir_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert collect_bfcl_scores(handler_key="nope", score_dir="score") == {}

    def test_reads_csv_files_under_model_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "score" / "gemma-4-12b-it"
        model_dir.mkdir(parents=True)
        with (model_dir / "multi_turn_base.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "accuracy"])
            writer.writeheader()
            writer.writerow({"category": "multi_turn_base", "accuracy": "0.55"})

        scores = collect_bfcl_scores(handler_key="gemma-4-12b-it", score_dir="score")
        assert "multi_turn_base" in scores
        assert scores["multi_turn_base"][0]["accuracy"] == "0.55"


class TestRunBfclEvalMissingCli:
    def test_raises_clear_error_when_bfcl_not_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentforge.eval.bfcl_runner.shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="bfcl CLI not found"):
            run_bfcl_eval(
                handler_key="m",
                local_model_path="p",
                adapter_dir="a",
                max_lora_rank=16,
                test_categories=["multi_turn_base"],
                regression_categories=["simple"],
                backend="vllm",
                run_name="test",
            )
