import pytest
import torch
from peft import LoraConfig as PeftLoraConfig
from transformers import BitsAndBytesConfig

from agentforge.model_utils import (
    assert_tools_render,
    build_bnb_config,
    build_lora_config,
    load_tokenizer,
)

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class TestBuildBnbConfig:
    def test_disabled_returns_none(self) -> None:
        assert build_bnb_config(enabled=False) is None

    def test_disabled_ignores_other_kwargs(self) -> None:
        # Even nonsense values shouldn't matter when disabled short-circuits.
        assert (
            build_bnb_config(
                enabled=False,
                load_in_4bit=False,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
            )
            is None
        )

    def test_enabled_returns_real_bnb_config(self) -> None:
        cfg = build_bnb_config(enabled=True)
        assert isinstance(cfg, BitsAndBytesConfig)

    def test_enabled_defaults(self) -> None:
        cfg = build_bnb_config(enabled=True)
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True

    def test_compute_dtype_string_resolved_to_real_torch_dtype(self) -> None:
        cfg = build_bnb_config(enabled=True, bnb_4bit_compute_dtype="bfloat16")
        assert cfg.bnb_4bit_compute_dtype is torch.bfloat16
        assert isinstance(cfg.bnb_4bit_compute_dtype, torch.dtype)

    def test_compute_dtype_float16(self) -> None:
        cfg = build_bnb_config(enabled=True, bnb_4bit_compute_dtype="float16")
        assert cfg.bnb_4bit_compute_dtype is torch.float16

    def test_invalid_compute_dtype_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="not a valid torch dtype"):
            build_bnb_config(enabled=True, bnb_4bit_compute_dtype="not_a_real_dtype")

    def test_custom_kwargs_pass_through(self) -> None:
        cfg = build_bnb_config(
            enabled=True,
            load_in_4bit=False,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype="float32",
            bnb_4bit_use_double_quant=False,
        )
        assert cfg.load_in_4bit is False
        assert cfg.bnb_4bit_quant_type == "fp4"
        assert cfg.bnb_4bit_compute_dtype is torch.float32
        assert cfg.bnb_4bit_use_double_quant is False


class TestBuildLoraConfig:
    def test_returns_real_peft_lora_config(self) -> None:
        cfg = build_lora_config(target_modules=DEFAULT_TARGET_MODULES)
        assert isinstance(cfg, PeftLoraConfig)

    def test_defaults(self) -> None:
        cfg = build_lora_config(target_modules=DEFAULT_TARGET_MODULES)
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.bias == "none"
        # peft stores target_modules internally as a set, so compare as sets.
        assert set(cfg.target_modules) == set(DEFAULT_TARGET_MODULES)

    def test_task_type_string_accepted_and_preserved(self) -> None:
        # peft 0.19.1's TaskType is a str-subclassing enum and LoraConfig
        # validates/accepts plain strings directly -- no TaskType[...]
        # conversion needed.
        cfg = build_lora_config(target_modules=DEFAULT_TARGET_MODULES, task_type="CAUSAL_LM")
        assert cfg.task_type == "CAUSAL_LM"

    def test_invalid_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="task type"):
            build_lora_config(target_modules=DEFAULT_TARGET_MODULES, task_type="NOT_A_TASK_TYPE")

    def test_custom_kwargs_pass_through(self) -> None:
        cfg = build_lora_config(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["gate_proj", "up_proj"],
            bias="all",
        )
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.1
        assert cfg.bias == "all"
        assert set(cfg.target_modules) == {"gate_proj", "up_proj"}


# --- Network/model-dependent tests below: these need a real tokenizer. ---

TOOL_AWARE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NON_TOOL_AWARE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.integration
class TestLoadTokenizer:
    def test_loads_real_tokenizer(self) -> None:
        tok = load_tokenizer(base_model=TOOL_AWARE_MODEL)
        assert tok is not None
        assert tok("hello world") is not None


@pytest.mark.integration
class TestAssertToolsRender:
    def test_happy_path_tool_aware_template(self) -> None:
        tok = load_tokenizer(base_model=TOOL_AWARE_MODEL)
        # Should not raise: Qwen2.5-Instruct's chat template renders tools.
        assert_tools_render(tok)

    def test_custom_sample_messages_and_tools(self) -> None:
        tok = load_tokenizer(base_model=TOOL_AWARE_MODEL)
        messages = [{"role": "user", "content": "book me a flight"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight.",
                    "parameters": {
                        "type": "object",
                        "properties": {"destination": {"type": "string"}},
                        "required": ["destination"],
                    },
                },
            }
        ]
        assert_tools_render(tok, sample_messages=messages, sample_tools=tools)

    def test_raises_when_template_silently_drops_tools(self) -> None:
        # TinyLlama's chat template has no notion of tools at all -- it
        # accepts (and ignores) the `tools` kwarg rather than erroring, so
        # this is exactly the silent-drop failure mode assert_tools_render
        # exists to catch.
        tok = load_tokenizer(base_model=NON_TOOL_AWARE_MODEL)
        with pytest.raises(AssertionError, match="get_weather"):
            assert_tools_render(tok)
