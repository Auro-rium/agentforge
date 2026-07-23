"""Small, focused helpers that wire together `transformers` / `peft` /
`bitsandbytes` config objects, plus a tokenizer sanity-check.

This module deliberately does NOT import from `agentforge.config` — it takes
plain keyword arguments whose names mirror the fields on the pydantic models
in `config.py`, so wiring config objects into these functions later is a thin
pass-through (e.g. `build_bnb_config(**quantization_config.model_dump())`).

Empirical notes (transformers==5.14.1, peft==0.19.1, bitsandbytes==0.49.2):

- `transformers.BitsAndBytesConfig(bnb_4bit_compute_dtype=...)` actually
  accepts a plain string like `"bfloat16"` in this version — it resolves the
  string internally via `getattr(torch, dtype_str)`. We still resolve the
  string to a real `torch.dtype` ourselves before constructing the config
  (see `_resolve_torch_dtype`) rather than relying on that internal
  behavior, so this module keeps working even if a future transformers
  release tightens the type check, and so an invalid dtype name fails with a
  clear `ValueError` instead of transformers' raw `AttributeError` on the
  `torch` module.
- `peft.LoraConfig(task_type=...)` accepts a plain string (`"CAUSAL_LM"`)
  directly in this version — `peft.utils.peft_types.TaskType` is a
  `str`-subclassing enum, and `LoraConfig.__post_init__` validates the
  string against the known task types itself, raising `ValueError` for
  anything invalid. No manual `TaskType[...]` conversion is required, so
  `build_lora_config` passes `task_type` straight through.
"""

from __future__ import annotations

import torch
import transformers
from peft import LoraConfig as PeftLoraConfig


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a string like "bfloat16" to a real `torch.dtype`.

    Raises a clear `ValueError` (rather than a raw `AttributeError` on the
    `torch` module) if `dtype_name` isn't a real torch dtype name.
    """
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"{dtype_name!r} is not a valid torch dtype name (e.g. 'bfloat16', 'float16')."
        )
    return dtype


def build_bnb_config(
    *,
    enabled: bool,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
) -> transformers.BitsAndBytesConfig | None:
    """Build a `transformers.BitsAndBytesConfig`, or `None` if quantization
    is disabled (meaning: full-precision/bf16 LoRA, no quantization).
    """
    if not enabled:
        return None
    return transformers.BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_resolve_torch_dtype(bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def build_lora_config(
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str],
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> PeftLoraConfig:
    """Build a `peft.LoraConfig`. Pass-through construction — see module
    docstring for why `task_type` doesn't need converting to `peft.TaskType`.
    """
    return PeftLoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )


def load_tokenizer(
    *, base_model: str, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerBase:
    """Thin wrapper around `AutoTokenizer.from_pretrained`."""
    return transformers.AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=trust_remote_code
    )


_DEFAULT_SAMPLE_MESSAGES: list[dict] = [
    {"role": "user", "content": "What is the weather in Paris?"},
]

_DEFAULT_SAMPLE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def assert_tools_render(
    tokenizer: transformers.PreTrainedTokenizerBase,
    *,
    sample_messages: list[dict] | None = None,
    sample_tools: list[dict] | None = None,
) -> None:
    """Fail-fast guard: assert that `tokenizer`'s chat template actually
    renders `tools` into the prompt, rather than silently dropping the
    `tools` kwarg (which some chat templates do if they were never written
    with tool-calling in mind).

    This is deliberately loose about *how* tools show up in the rendered
    string — different chat templates format tool schemas very differently
    (XML tags, JSON blocks, etc.) — and only checks that the tool's
    function `name` appears somewhere in the output.
    """
    messages = sample_messages if sample_messages is not None else _DEFAULT_SAMPLE_MESSAGES
    tools = sample_tools if sample_tools is not None else _DEFAULT_SAMPLE_TOOLS

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )

    missing = [
        name
        for tool in tools
        if (name := tool.get("function", {}).get("name")) and name not in rendered
    ]
    if missing:
        raise AssertionError(
            "tokenizer.apply_chat_template() did not render the following tool name(s) "
            f"anywhere in its output: {missing!r}. This usually means the tokenizer's "
            "chat template silently ignores the `tools` kwarg — double check that the "
            "base model's chat template actually supports tool calling.\n"
            f"Rendered output was:\n{rendered!r}"
        )
