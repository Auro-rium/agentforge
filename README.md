# AgentForge

AgentForge instruction-tunes `google/gemma-4-12B-it` via LoRA/QLoRA (TRL's
`SFTTrainer`) for **multi-turn agentic tool-call reliability**.

## Why this project, specifically

Before committing to this scope, two independent research passes checked
whether it was worth building. Both converged on the same finding: a
generic "fine-tune a model for tool calling" project would have been
redundant — `gemma-4-12B-it` already ships native tool-call tokens, and the
niche already has 50+ community fine-tunes within weeks of the base model's
release. What *isn't* already solved: Gemma 4-12B-it has a documented bug
([HF discussion #28](https://huggingface.co/google/gemma-4-12B-it/discussions/28))
where it loses track of its own prior tool outputs across a multi-turn
conversation, patched by Google via a chat-template workaround, not a
retrain. Multi-turn tool-call reliability is the field's actual unsolved
problem — even leading small models drop from ~88% single-turn accuracy to
~55% multi-turn on BFCL.

So the project is scoped narrowly: **fix Gemma-4-12B-it's multi-turn
tool-context reliability through training**, not add tool-calling (already
present) or compete on single-turn benchmarks (already crowded).

## Data

Five public datasets, normalized into one canonical schema
(`src/agentforge/data/schema.py` — OpenAI-style `messages` + `tools`, the
shape TRL's `SFTTrainer` expects natively):

| Source | Role | Sampling weight |
|---|---|---|
| `internlm/Agent-FLAN` | **Headline dataset** — ReAct (Thought/Action/Observation) multi-turn trajectories, parsed into structured tool calls | 3.0 (upweighted) |
| `glaiveai/glaive-function-calling-v2` | Single-turn schema-grounded tool calls | 1.0 |
| `NousResearch/hermes-function-calling-v1` | Structured function-calling + JSON-mode | 1.0 |
| `Team-ACE/ToolACE` | Multi-turn, diverse tool-use trajectories | 1.0 |
| `Salesforce/xlam-function-calling-60k` | 60k function-calling examples (gated — needs `HF_TOKEN`) | 0.5 |

Each source has its own normalizer under `src/agentforge/data/normalizers/`,
tested against hand-written fixtures in each dataset's real native format
(no network needed for tests). `data/build_manifest.py` runs all five,
dedupes across sources, carves a stratified held-out split, applies the
weights above, and writes `data/manifest.jsonl` + `data/holdout.jsonl` +
`data/manifest_stats.json`.

## Compute: everything real runs on a rented AWS GPU instance

This repo is developed and tested locally (CPU-only, no real datasets), but
**dataset fetching and training always run on a rented AWS GPU instance**,
never on a local machine:

```bash
export AWS_KEY_NAME=...              # existing EC2 key pair
export AWS_SECURITY_GROUP_ID=...
export AWS_IAM_INSTANCE_PROFILE=...  # needs s3:PutObject on the bucket below
export AGENTFORGE_S3_BUCKET=s3://your-bucket/agentforge
export AGENTFORGE_GIT_REMOTE=https://github.com/Auro-rium/agentforge.git
export HF_TOKEN=...                  # dataset-read (xlam) + repo-write (auro-rirum) scopes

bash scripts/aws/launch_instance.sh
```

This provisions a GPU instance (`g5.2xlarge` by default — one A10G, 24GB,
enough for `gemma-4-12B-it` QLoRA) and runs, unattended, entirely on that
instance: dataset download → manifest build → training → fast dev-loop eval
→ publish the adapter to
[huggingface.co/auro-rirum](https://huggingface.co/auro-rirum) → sync
checkpoints/reports to S3. Tail progress with:

```bash
ssh -i <your-key>.pem ubuntu@<instance-ip> 'tail -f /var/log/agentforge-bootstrap.log'
```

BFCL and τ²-bench (the primary eval targets — see below) aren't run
automatically, since they need heavier deps (`vllm`) and, for τ²-bench, a
real integration spike that hasn't happened yet. Run them manually once
training completes:

```bash
scripts/run_bfcl.sh <adapter_dir>
```

## Local development

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/         # fast suite: schema, normalizers, config, model_utils, eval scorers
.venv/bin/python -m pytest tests/ -m slow # + a real CPU training smoke run (downloads a small model)
```

The `slow`-marked smoke test (`tests/test_train_smoke.py`) validates the
full `SFTTrainer`/`LoraConfig`/dataset wiring against a tiny real model on
CPU before any GPU time is rented — the single most valuable pre-flight
check in the project, since it catches TRL/transformers/peft API drift for
free.

## Evaluation

Post-pivot, the headline metrics are **not** overall BFCL — that benchmark
weights 70% of its score toward agentic/multi-turn categories this recipe
doesn't uniquely target, and the base model is already reasonably
competitive on plain single-turn function calling. Instead:

- **BFCL v4 multi-turn subsets** (`multi_turn_base`, `multi_turn_miss_func`,
  `multi_turn_miss_param`, `multi_turn_long_context`) — the primary signal.
  Single-turn categories are tracked only as a regression check.
- **τ²-bench** — a second, more realistic dual-control multi-turn signal.
  `src/agentforge/eval/tau2_runner.py` is an intentionally honest
  placeholder until a real integration spike happens (see its docstring).
- **Fast dev-loop scorer** (`scripts/dev_eval.sh`) — JSON validity,
  function-name/argument match, and a task-success proxy against the
  held-out set, cheap enough to run between GPU-rental sessions.

## Project structure

```
configs/            YAML training recipes (gemma4-12b-qlora.yaml, smoke-cpu-tiny.yaml)
src/agentforge/
  data/              canonical schema, per-source normalizers, manifest pipeline
  eval/              tool-call parsers, dev-loop scorer, BFCL/τ²-bench runners
  config.py          pydantic YAML config loader
  model_utils.py     BitsAndBytesConfig/LoraConfig builders, tools-render guard
  train.py           TRL SFTTrainer entrypoint
  infer.py           adapter-mode / merged-mode inference
  merge_adapter.py   LoRA -> standalone merged model
  publish_hf.py      push to huggingface.co/auro-rirum
scripts/             thin CLI wrappers around the above
scripts/aws/         EC2 provisioning + unattended bootstrap-and-train
```
