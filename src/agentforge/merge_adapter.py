"""Merge a trained LoRA adapter into its base model, producing a standalone
HF model directory.

QLoRA constraint: this always reloads the base model in bf16 (never
quantized) before merging -- `bitsandbytes.nn.Linear4bit` layers aren't
mergeable, so merging into a 4-bit-loaded model isn't possible regardless of
what precision the adapter was originally trained under. This needs real
memory (~half the model's full-precision footprint, e.g. ~24GB for a 12B
model in bf16), even if training happened under 4-bit QLoRA on a smaller
GPU -- run this step on a box with enough RAM/VRAM for that.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from agentforge.model_utils import load_tokenizer


def merge_adapter(*, base_model: str, adapter_dir: str, output_dir: str, trust_remote_code: bool = False) -> None:
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir)

    tokenizer = load_tokenizer(base_model=base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    merge_adapter(base_model=args.base_model, adapter_dir=args.adapter_dir, output_dir=args.output_dir)
    print(f"Merged model written to {args.output_dir}")


if __name__ == "__main__":
    main()
