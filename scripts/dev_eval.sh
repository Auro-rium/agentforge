#!/usr/bin/env bash
# Fast offline dev-loop eval against the held-out set, adapter-mode (no
# merge needed -- dev_eval.py loads the base model + adapter directly).
# Usage: scripts/dev_eval.sh <adapter_dir> [run_name]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER_DIR="${1:?Usage: scripts/dev_eval.sh <adapter_dir> [run_name]}"
RUN_NAME="${2:-dev_eval}"

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "error: adapter dir not found: ${ADAPTER_DIR}" >&2
  exit 1
fi

"${PYTHON}" -m agentforge.eval.dev_eval --adapter-dir "${ADAPTER_DIR}" --run-name "${RUN_NAME}"
