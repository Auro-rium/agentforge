#!/usr/bin/env bash
# Produces a standalone merged model (for serving stacks without LoRA
# support) AND separately confirms quality via the adapter-mode dev-loop
# scorer -- two independent conveniences bundled for the common
# "prepare for serving + sanity check" workflow. Note the eval step
# evaluates the *adapter* (base + LoRA), not the merged output -- that's
# deliberate (adapter-mode needs no merge at all, see scripts/dev_eval.sh),
# not a bug; the merge is purely for producing a standalone artifact here.
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
