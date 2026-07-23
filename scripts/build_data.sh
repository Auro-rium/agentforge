#!/usr/bin/env bash
# Fetches the 5 source HF datasets and builds data/manifest.jsonl + data/holdout.jsonl.
# Intended to run on the AWS training instance, not the local dev machine --
# it downloads real datasets (including the gated Salesforce/xlam-function-calling-60k,
# which needs HF_TOKEN set and its terms accepted on huggingface.co first).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${HF_TOKEN:?Set HF_TOKEN before fetching Salesforce/xlam-function-calling-60k (gated dataset)}"

PYTHON="${AGENTFORGE_PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: ${PYTHON} not found -- run this from the repo root after 'uv venv .venv && uv pip install -e .[eval,dev]'" >&2
  exit 1
fi

mkdir -p data/normalized

"${PYTHON}" -m agentforge.data.build_manifest "$@"

echo "Manifest build complete:"
"${PYTHON}" -c "
import json
with open('data/manifest_stats.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
