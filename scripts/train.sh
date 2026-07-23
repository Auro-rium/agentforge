#!/usr/bin/env bash
# Runs real training via TRL's SFTTrainer against a config-driven recipe.
# Usage: scripts/train.sh [configs/gemma4-12b-qlora.yaml]
# Intended to run on the AWS GPU instance, not the local dev machine.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:-configs/gemma4-12b-qlora.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "error: config file not found: ${CONFIG}" >&2
  exit 1
fi

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

if [[ ! -f data/manifest.jsonl ]]; then
  echo "error: data/manifest.jsonl not found -- run scripts/build_data.sh first" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
else
  GPU_COUNT=0
fi

if [[ "${GPU_COUNT}" -gt 1 ]]; then
  echo "Detected ${GPU_COUNT} GPUs, launching via accelerate."
  "${PYTHON}" -m accelerate.commands.launch -m agentforge.train --config "${CONFIG}"
elif [[ "${GPU_COUNT}" -eq 1 ]]; then
  echo "Detected 1 GPU."
  "${PYTHON}" -m agentforge.train --config "${CONFIG}"
else
  echo "warning: no GPU detected -- this is only sane for configs/smoke-cpu-tiny.yaml" >&2
  "${PYTHON}" -m agentforge.train --config "${CONFIG}"
fi
