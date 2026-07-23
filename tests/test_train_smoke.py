"""CPU smoke test: validates the SFTTrainer + LoraConfig + dataset wiring in
train.py actually runs against the installed transformers/trl/peft versions,
catching API drift before any real (rented) GPU time is spent. Network- and
CPU-time-heavy (downloads Qwen2.5-0.5B-Instruct, runs a few real training
steps) -- marked slow, excluded from the default fast test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentforge.config import AgentForgeConfig
from agentforge.data.schema import FunctionCall, Message, Row, ToolCall
from agentforge.train import run_training

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _synthetic_row(i: int) -> Row:
    return Row(
        id=f"smoke_{i}",
        source="glaive",
        messages=[
            Message(role="system", content="You are a helpful assistant with access to tools."),
            Message(role="user", content=f"What is the weather in city {i}?"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_0",
                        function=FunctionCall(name="get_weather", arguments=json.dumps({"city": f"city{i}"})),
                    )
                ],
            ),
            Message(role="tool", content="18C", tool_call_id="call_0", name="get_weather"),
            Message(role="assistant", content=f"It's 18C in city {i}."),
        ],
        tools=[],
    )


def _write_jsonl(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json())
            f.write("\n")


@pytest.mark.slow
class TestTrainSmoke:
    def test_cpu_smoke_run_completes_and_saves_checkpoint(self, tmp_path: Path) -> None:
        cfg = AgentForgeConfig.from_yaml(CONFIGS_DIR / "smoke-cpu-tiny.yaml")

        manifest_path = tmp_path / "manifest.jsonl"
        holdout_path = tmp_path / "holdout.jsonl"
        output_dir = tmp_path / "outputs" / "smoke-cpu-tiny"

        _write_jsonl([_synthetic_row(i) for i in range(5)], manifest_path)
        _write_jsonl([_synthetic_row(i) for i in range(5, 7)], holdout_path)

        cfg.data.manifest_path = str(manifest_path)
        cfg.data.holdout_path = str(holdout_path)
        cfg.training.output_dir = str(output_dir)
        cfg.training.num_train_epochs = 1
        cfg.training.per_device_train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.eval_steps = 1
        cfg.training.save_steps = 1
        cfg.training.logging_steps = 1

        run_training(cfg)

        assert output_dir.exists()
        saved_files = list(output_dir.iterdir())
        assert len(saved_files) > 0, f"expected checkpoint files in {output_dir}, found none"
