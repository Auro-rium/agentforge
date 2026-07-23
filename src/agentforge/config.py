"""Pydantic-based YAML config loader for training runs.

`AgentForgeConfig.from_yaml(path)` reads a YAML training config (e.g.
`configs/gemma4-12b-qlora.yaml`) and validates it into a nested pydantic model
tree, so `train.py` gets a fully-typed, fail-fast config object instead of a
raw dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    base_model: str
    trust_remote_code: bool = False
    chat_template_path: str | None = None
    attn_implementation: str = "sdpa"


class QuantizationConfig(BaseModel):
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"


class DataConfig(BaseModel):
    manifest_path: str = "data/manifest.jsonl"
    holdout_path: str = "data/holdout.jsonl"
    sources: dict[str, float] = Field(default_factory=dict)
    max_examples_per_source: int | None = None
    shuffle: bool = True


class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_length: int = 4096
    packing: bool = False
    assistant_only_loss: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 200
    report_to: list[str] = Field(default_factory=list)


class DevHoldoutConfig(BaseModel):
    enabled: bool = True


class BfclConfig(BaseModel):
    enabled: bool = False
    handler_key: str = "gemma-4-12b-it"
    test_categories: list[str] = Field(
        default_factory=lambda: [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ]
    )
    regression_categories: list[str] = Field(
        default_factory=lambda: ["simple", "multiple", "parallel"]
    )
    backend: Literal["vllm", "sglang"] = "vllm"


class Tau2BenchConfig(BaseModel):
    enabled: bool = False
    repo: str = "https://github.com/sierra-research/tau2-bench"
    domains: list[str] | None = None


class EvalConfig(BaseModel):
    dev_holdout: DevHoldoutConfig = Field(default_factory=DevHoldoutConfig)
    bfcl: BfclConfig = Field(default_factory=BfclConfig)
    tau2_bench: Tau2BenchConfig = Field(default_factory=Tau2BenchConfig)


class AgentForgeConfig(BaseModel):
    run_name: str
    seed: int = 42
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentForgeConfig:
        """Load and validate a training config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)
