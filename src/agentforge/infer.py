"""Load a trained LoRA adapter (adapter-mode, default) or a merged model
(merged-mode) and run a chat/tool-calling completion.

Adapter-mode is preferred by default -- it's what scripts/run_bfcl.sh's
`bfcl generate --enable-lora` expects, and it's smaller on disk / keeps
multiple task adapters swappable. Use merge_adapter.py first if you need a
standalone merged model directory instead (e.g. for a serving stack without
clean LoRA-adapter support).
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from agentforge.model_utils import build_bnb_config, load_tokenizer


def load_adapter_model(
    *,
    base_model: str,
    adapter_dir: str,
    quantization_enabled: bool,
    bnb_kwargs: dict | None = None,
    trust_remote_code: bool = False,
):
    """Load `base_model` (quantized if `quantization_enabled`, matching
    whatever precision the adapter was trained under) plus the LoRA adapter
    at `adapter_dir`.
    """
    bnb_config = build_bnb_config(enabled=quantization_enabled, **(bnb_kwargs or {}))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if bnb_config is None else None,
        trust_remote_code=trust_remote_code,
    )
    return PeftModel.from_pretrained(model, adapter_dir)


def load_merged_model(*, merged_dir: str, trust_remote_code: bool = False):
    """Load a standalone merged model directory (see merge_adapter.py).
    Never quantized -- merging into a 4-bit-loaded model isn't supported
    (bitsandbytes.nn.Linear4bit layers aren't mergeable), so a merged
    checkpoint is always full/half precision on disk already.
    """
    return AutoModelForCausalLM.from_pretrained(
        merged_dir, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )


def generate_completion(
    model,
    tokenizer,
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int = 512,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--quantized", action="store_true", help="Load base model in 4-bit (QLoRA-trained adapters)")
    parser.add_argument("--messages", required=True, help="JSON list of {role, content} messages")
    parser.add_argument("--tools", default=None, help="JSON list of tool schemas")
    args = parser.parse_args()

    tokenizer = load_tokenizer(base_model=args.base_model)
    model = load_adapter_model(
        base_model=args.base_model, adapter_dir=args.adapter_dir, quantization_enabled=args.quantized
    )
    messages = json.loads(args.messages)
    tools = json.loads(args.tools) if args.tools else None
    completion = generate_completion(model, tokenizer, messages=messages, tools=tools)
    print(completion)


if __name__ == "__main__":
    main()
