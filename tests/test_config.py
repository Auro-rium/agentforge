from pathlib import Path

import pytest
from pydantic import ValidationError

from agentforge.config import (
    AgentForgeConfig,
    BfclConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestFromYamlRealConfigs:
    def test_main_config_loads(self) -> None:
        cfg = AgentForgeConfig.from_yaml(CONFIGS_DIR / "gemma4-12b-qlora.yaml")
        assert cfg.run_name == "gemma4-12b-qlora-agent-v1"
        assert cfg.model.base_model == "google/gemma-4-12B-it"
        assert cfg.quantization.enabled is True
        assert cfg.lora.target_modules == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert cfg.data.sources["agent_flan"] == 3.0
        assert cfg.training.report_to == ["wandb"]
        assert cfg.eval.bfcl.enabled is False
        assert cfg.eval.tau2_bench.domains is None

    def test_smoke_config_loads(self) -> None:
        cfg = AgentForgeConfig.from_yaml(CONFIGS_DIR / "smoke-cpu-tiny.yaml")
        assert cfg.run_name == "smoke-cpu-tiny"
        assert cfg.quantization.enabled is False
        assert cfg.data.max_examples_per_source == 5
        assert cfg.training.max_length == 128
        assert cfg.training.num_train_epochs == 1
        assert cfg.training.report_to == []
        assert cfg.training.output_dir == "outputs/smoke-cpu-tiny"


class TestBadEnumValues:
    def test_bad_lora_bias_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoraConfig(bias="not_a_real_option")

    def test_bad_bfcl_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BfclConfig(backend="not_vllm_or_sglang")

    def test_bad_quant_type_rejected(self) -> None:
        from agentforge.config import QuantizationConfig

        with pytest.raises(ValidationError):
            QuantizationConfig(bnb_4bit_quant_type="int8")

    def test_bad_save_strategy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(output_dir="x", save_strategy="whenever")


class TestOptionalFieldDefaults:
    def test_chat_template_path_defaults_none(self) -> None:
        cfg = ModelConfig(base_model="gpt2")
        assert cfg.chat_template_path is None

    def test_max_examples_per_source_defaults_none(self) -> None:
        from agentforge.config import DataConfig

        cfg = DataConfig()
        assert cfg.max_examples_per_source is None

    def test_tau2_domains_defaults_none(self) -> None:
        from agentforge.config import Tau2BenchConfig

        cfg = Tau2BenchConfig()
        assert cfg.domains is None


class TestFullRoundTrip:
    def _minimal_config_dict(self) -> dict:
        return {
            "run_name": "unit-test-run",
            "seed": 7,
            "model": {"base_model": "sshleifer/tiny-gpt2"},
            "quantization": {},
            "lora": {},
            "data": {
                "sources": {"agent_flan": 3.0, "glaive": 1.0},
            },
            "training": {"output_dir": "outputs/unit-test-run"},
            "eval": {},
        }

    def test_minimal_dict_round_trip(self) -> None:
        raw = self._minimal_config_dict()
        cfg = AgentForgeConfig(**raw)

        assert cfg.run_name == "unit-test-run"
        assert cfg.seed == 7
        assert cfg.model.base_model == "sshleifer/tiny-gpt2"
        assert cfg.model.chat_template_path is None
        assert cfg.data.sources["agent_flan"] == 3.0
        assert cfg.lora.r == 16
        assert cfg.training.output_dir == "outputs/unit-test-run"
        assert cfg.eval.dev_holdout.enabled is True
        assert cfg.eval.bfcl.enabled is False

    def test_missing_required_field_rejected(self) -> None:
        raw = self._minimal_config_dict()
        del raw["model"]
        with pytest.raises(ValidationError):
            AgentForgeConfig(**raw)
