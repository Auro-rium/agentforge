#!/usr/bin/env bash
# Merges a trained LoRA adapter into a full-precision base model, then runs the
# fast offline dev-loop scorer against the held-out set. Run on the training
# instance (or any box with the base model + adapter available).
# Usage: scripts/merge_and_eval.sh <adapter_dir> [merged_output_dir]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER_DIR="${1:?Usage: scripts/merge_and_eval.sh <adapter_dir> [merged_output_dir]}"
MERGED_DIR="${2:-${ADAPTER_DIR%/}-merged}"

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "error: adapter dir not found: ${ADAPTER_DIR}" >&2
  exit 1
fi

echo "Merging adapter ${ADAPTER_DIR} -> ${MERGED_DIR} (reloads base in bf16, no quantization)..."
"${PYTHON}" -m agentforge.merge_adapter --adapter-dir "${ADAPTER_DIR}" --output-dir "${MERGED_DIR}"

echo "Running fast dev-loop scorer against the held-out set..."
"${PYTHON}" -m agentforge.eval.dev_eval --adapter-dir "${ADAPTER_DIR}"
