#!/usr/bin/env bash
# Runs ON the AWS GPU instance -- never on the local dev machine. Either invoked
# automatically as EC2 user-data (launch_instance.sh prepends the required
# exports before this script's body) or manually over SSH after exporting the
# same env vars yourself.
#
# Does, in order, all on the instance: clone the repo -> install deps ->
# fetch the 5 source datasets + build the training manifest -> run real
# training -> fast dev-loop eval -> publish the adapter to
# https://huggingface.co/auro-rirum -> sync checkpoints/reports back to S3.
# Nothing here touches the local machine that launched the instance.
#
# HF_TOKEN is dual-purpose: it authenticates the gated
# Salesforce/xlam-function-calling-60k dataset download AND the final
# publish-to-Hub step, so it needs both dataset-read and repo-write scopes
# on the auro-rirum account.

set -euo pipefail

: "${AGENTFORGE_S3_BUCKET:?Set AGENTFORGE_S3_BUCKET, e.g. s3://my-bucket/agentforge}"
: "${AGENTFORGE_GIT_REMOTE:?Set AGENTFORGE_GIT_REMOTE to this repos git URL}"
: "${HF_TOKEN:?Set HF_TOKEN -- needs both dataset-read (xlam) and repo-write (auro-rirum) scopes}"

AGENTFORGE_GIT_REF="${AGENTFORGE_GIT_REF:-main}"
AGENTFORGE_TRAIN_CONFIG="${AGENTFORGE_TRAIN_CONFIG:-configs/gemma4-12b-qlora.yaml}"
AGENTFORGE_HF_REPO_NAME="${AGENTFORGE_HF_REPO_NAME:-gemma4-12b-agentforge}"
WORKDIR="${AGENTFORGE_WORKDIR:-/home/ubuntu/agentforge}"
LOGFILE="/var/log/agentforge-bootstrap.log"

exec > >(tee -a "${LOGFILE}") 2>&1
echo "=== agentforge bootstrap starting at $(date -u --iso-8601=seconds) ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: no GPU detected on this instance (nvidia-smi not found) -- refusing to proceed" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

if [[ -d "${WORKDIR}/.git" ]]; then
  echo "Repo already present at ${WORKDIR}, pulling latest ${AGENTFORGE_GIT_REF}..."
  git -C "${WORKDIR}" fetch origin "${AGENTFORGE_GIT_REF}"
  git -C "${WORKDIR}" checkout "${AGENTFORGE_GIT_REF}"
  git -C "${WORKDIR}" pull origin "${AGENTFORGE_GIT_REF}"
else
  echo "Cloning ${AGENTFORGE_GIT_REMOTE} (ref ${AGENTFORGE_GIT_REF}) into ${WORKDIR}..."
  git clone --branch "${AGENTFORGE_GIT_REF}" "${AGENTFORGE_GIT_REMOTE}" "${WORKDIR}"
fi
cd "${WORKDIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "Creating venv and installing dependencies (including [eval] extras for BFCL/vllm)..."
uv venv .venv
uv pip install --python .venv/bin/python -e ".[eval,dev]"

export HF_TOKEN
echo "=== Fetching source datasets + building training manifest (on-instance only) ==="
bash scripts/build_data.sh

echo "=== Starting training: ${AGENTFORGE_TRAIN_CONFIG} ==="
bash scripts/train.sh "${AGENTFORGE_TRAIN_CONFIG}"

OUTPUT_DIR="$(.venv/bin/python -c "
from agentforge.config import AgentForgeConfig
print(AgentForgeConfig.from_yaml('${AGENTFORGE_TRAIN_CONFIG}').training.output_dir)
")"

echo "=== Fast dev-loop eval against the held-out set (adapter-mode, no merge needed) ==="
bash scripts/dev_eval.sh "${OUTPUT_DIR}" || echo "warning: dev-loop eval failed, continuing to publish anyway"

echo "=== Publishing adapter to https://huggingface.co/auro-rirum/${AGENTFORGE_HF_REPO_NAME} ==="
DEV_EVAL_SUMMARY="reports/dev_eval/dev_eval/summary.json"
if [[ -f "${DEV_EVAL_SUMMARY}" ]]; then
  bash scripts/publish_hf.sh "${OUTPUT_DIR}" "${AGENTFORGE_HF_REPO_NAME}" adapter "${DEV_EVAL_SUMMARY}"
else
  bash scripts/publish_hf.sh "${OUTPUT_DIR}" "${AGENTFORGE_HF_REPO_NAME}" adapter
fi

echo "=== Syncing artifacts to ${AGENTFORGE_S3_BUCKET} ==="
aws s3 sync data/manifest_stats.json "${AGENTFORGE_S3_BUCKET}/manifest_stats.json" 2>/dev/null \
  || aws s3 cp data/manifest_stats.json "${AGENTFORGE_S3_BUCKET}/manifest_stats.json"
aws s3 sync outputs/ "${AGENTFORGE_S3_BUCKET}/outputs/"
aws s3 sync reports/ "${AGENTFORGE_S3_BUCKET}/reports/"

echo "=== agentforge bootstrap finished at $(date -u --iso-8601=seconds) ==="
echo "Checkpoints and reports are in ${AGENTFORGE_S3_BUCKET}."
echo "Adapter published to https://huggingface.co/auro-rirum/${AGENTFORGE_HF_REPO_NAME}."
echo "Not run automatically (need bfcl-eval/vllm and/or a real tau2-bench spike -- see docs/plan):"
echo "  scripts/run_bfcl.sh ${OUTPUT_DIR}"
echo "  scripts/merge_and_eval.sh ${OUTPUT_DIR}   # for a standalone merged model + sanity eval"
echo "This instance is still running -- terminate it yourself once you've confirmed the sync landed."
