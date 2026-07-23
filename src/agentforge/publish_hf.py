"""Publish a trained checkpoint (adapter or merged model) to the Hugging Face
Hub under the project owner's account (https://huggingface.co/auro-rirum).

Two publish modes:
  - adapter (default): pushes the LoRA adapter directory as-is -- small,
    fast, requires the base model + peft to reload. This is what
    scripts/run_bfcl.sh's `--enable-lora` path and infer.py's adapter-mode
    both expect.
  - merged: pushes a standalone merged model directory (produced by
    merge_adapter.py) -- larger, self-contained, no peft needed to load.

Requires `HF_TOKEN` (a Hugging Face access token with write access to the
auro-rirum account) set in the environment. Intended to run as the final
step of a real training run on the AWS instance (see
scripts/aws/bootstrap_and_train.sh), after training + eval, not as a
standalone local action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

HF_OWNER = "auro-rirum"

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: {base_model}
tags:
- agentforge
- lora
- tool-calling
- agentic
- multi-turn
---

# {repo_name}

LoRA adapter fine-tuning `{base_model}` for multi-turn agentic tool-call
reliability, produced by [agentforge](https://github.com/Auro-rium/agentforge).

Trained primarily on `internlm/Agent-FLAN`'s ReAct (Thought/Action/Observation)
trajectories, with `glaiveai/glaive-function-calling-v2`,
`NousResearch/hermes-function-calling-v1`, `Team-ACE/ToolACE`, and
`Salesforce/xlam-function-calling-60k` in a supporting role for
schema-grounded single-call argument correctness.

Targets improving `{base_model}`'s multi-turn tool-context reliability,
evaluated on BFCL v4's multi-turn subsets and τ²-bench.

## Usage (adapter mode)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{hf_repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

{metrics_section}
"""


def _load_metrics_section(metrics_path: str | None) -> str:
    if not metrics_path or not Path(metrics_path).exists():
        return ""
    metrics = json.loads(Path(metrics_path).read_text())
    lines = ["## Metrics", "", "```json", json.dumps(metrics, indent=2), "```"]
    return "\n".join(lines)


def publish_to_hub(
    *,
    local_dir: str,
    repo_name: str,
    base_model: str,
    mode: str = "adapter",
    metrics_path: str | None = None,
    private: bool = False,
) -> str:
    """Create (if needed) `auro-rirum/<repo_name>` and push `local_dir`'s
    contents to it, along with an auto-generated model card. Returns the
    full repo id.
    """
    if mode not in ("adapter", "merged"):
        raise ValueError(f"mode must be 'adapter' or 'merged', got {mode!r}")

    repo_id = f"{HF_OWNER}/{repo_name}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    card = MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name,
        base_model=base_model,
        hf_repo_id=repo_id,
        metrics_section=_load_metrics_section(metrics_path),
    )
    (Path(local_dir) / "README.md").write_text(card)

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        commit_message=f"Upload {mode} checkpoint via agentforge/publish_hf.py",
    )
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir", required=True, help="Adapter or merged model directory to upload"
    )
    parser.add_argument(
        "--repo-name", required=True, help="Repo name under auro-rirum, e.g. gemma4-12b-agentforge"
    )
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
    parser.add_argument("--mode", choices=["adapter", "merged"], default="adapter")
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Optional path to a JSON metrics file to embed in the model card",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    repo_id = publish_to_hub(
        local_dir=args.local_dir,
        repo_name=args.repo_name,
        base_model=args.base_model,
        mode=args.mode,
        metrics_path=args.metrics_path,
        private=args.private,
    )
    print(f"Published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
