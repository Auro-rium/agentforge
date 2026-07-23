"""Training entrypoint: config -> tokenizer -> BitsAndBytesConfig/LoraConfig
-> TRL SFTTrainer -> checkpoint.

Usage (single GPU):
    python -m agentforge.train --config configs/gemma4-12b-qlora.yaml
Usage (multi-GPU):
    accelerate launch -m agentforge.train --config configs/gemma4-12b-qlora.yaml

Intended to run on the AWS training instance (see scripts/train.sh /
scripts/aws/bootstrap_and_train.sh), not the local dev machine.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from agentforge.config import AgentForgeConfig
from agentforge.model_utils import (
    assert_tools_render,
    build_bnb_config,
    build_lora_config,
    load_tokenizer,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_init_kwargs(cfg: AgentForgeConfig) -> dict:
    kwargs: dict = {
        "dtype": "bfloat16",
        "attn_implementation": cfg.model.attn_implementation,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if torch.cuda.is_available():
        # Multi-GPU QLoRA footgun: under `accelerate launch` (one process per
        # GPU, DDP), a quantized model loaded with no device_map gets placed
        # the same way in every process instead of each process owning its
        # own GPU -- causing a crash or silent single-GPU-only training
        # rather than real data-parallel scaling across all of them. Explicit
        # per-process placement (the standard bitsandbytes+accelerate DDP
        # fix) makes both the single-GPU and multi-GPU cases correct:
        # PartialState().process_index is 0 (and harmless) for a single
        # process, and each rank's own index under `accelerate launch`.
        # Skipped when CUDA isn't available (the CPU smoke test) since
        # cuda:0 wouldn't exist there.
        from accelerate import PartialState

        kwargs["device_map"] = {"": PartialState().process_index}
    return kwargs


def build_sft_config(cfg: AgentForgeConfig) -> SFTConfig:
    t = cfg.training
    return SFTConfig(
        output_dir=t.output_dir,
        max_length=t.max_length,
        packing=t.packing,
        assistant_only_loss=t.assistant_only_loss,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        gradient_checkpointing=t.gradient_checkpointing,
        bf16=t.bf16,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        report_to=t.report_to,
        seed=cfg.seed,
        model_init_kwargs=_model_init_kwargs(cfg),
    )


def run_training(cfg: AgentForgeConfig) -> None:
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(
        base_model=cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if cfg.model.chat_template_path:
        with open(cfg.model.chat_template_path) as f:
            tokenizer.chat_template = f.read()
    # Fail fast: if the base model's chat template silently drops `tools`,
    # every training example's tool-call turns would be invisible to the
    # model and we'd only discover it after burning real GPU time.
    assert_tools_render(tokenizer)

    # build_bnb_config already returns None internally when enabled=False.
    bnb_config = build_bnb_config(**cfg.quantization.model_dump())
    lora_config = build_lora_config(**cfg.lora.model_dump())

    train_dataset = load_dataset("json", data_files=cfg.data.manifest_path, split="train")
    eval_dataset = load_dataset("json", data_files=cfg.data.holdout_path, split="train")

    sft_args = build_sft_config(cfg)

    trainer = SFTTrainer(
        model=cfg.model.base_model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        quantization_config=bnb_config,
    )
    trainer.train()
    trainer.save_model(cfg.training.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a training YAML config")
    args = parser.parse_args()

    cfg = AgentForgeConfig.from_yaml(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
