#!/usr/bin/env bash
# Publishes a trained adapter (or merged model) to
# https://huggingface.co/auro-rirum. Requires HF_TOKEN with write access to
# that account.
# Usage: scripts/publish_hf.sh <local_dir> <repo_name> [adapter|merged] [metrics_path]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOCAL_DIR="${1:?Usage: scripts/publish_hf.sh <local_dir> <repo_name> [adapter|merged] [metrics_path]}"
REPO_NAME="${2:?Usage: scripts/publish_hf.sh <local_dir> <repo_name> [adapter|merged] [metrics_path]}"
MODE="${3:-adapter}"
METRICS_PATH="${4:-}"

: "${HF_TOKEN:?Set HF_TOKEN (a Hugging Face access token with write access to the auro-rirum account)}"

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

ARGS=(--local-dir "${LOCAL_DIR}" --repo-name "${REPO_NAME}" --mode "${MODE}")
if [[ -n "${METRICS_PATH}" ]]; then
  ARGS+=(--metrics-path "${METRICS_PATH}")
fi

"${PYTHON}" -m agentforge.publish_hf "${ARGS[@]}"
