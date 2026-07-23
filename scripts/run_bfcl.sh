#!/usr/bin/env bash
# Runs BFCL v4's multi-turn subsets (the project's primary eval target, per the
# post-pivot scope -- see docs/plan) against a trained adapter, via vLLM.
# Requires the `eval` extra (bfcl-eval + vllm) installed: uv pip install -e ".[eval]"
# Usage: scripts/run_bfcl.sh <adapter_dir> [handler_key]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER_DIR="${1:?Usage: scripts/run_bfcl.sh <adapter_dir> [handler_key]}"
HANDLER_KEY="${2:-gemma-4-12b-it}"

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

if ! "${PYTHON}" -c "import bfcl_eval" 2>/dev/null; then
  echo "error: bfcl-eval not installed -- run: uv pip install --python ${PYTHON} -e '.[eval]'" >&2
  exit 1
fi

"${PYTHON}" -m agentforge.eval.bfcl_runner --adapter-dir "${ADAPTER_DIR}" --handler-key "${HANDLER_KEY}"
