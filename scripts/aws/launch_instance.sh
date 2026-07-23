#!/usr/bin/env bash
# Provisions a single AWS GPU instance for agentforge data-build + training.
#
# By design this does NOT run anything on the local machine beyond the AWS
# API calls to create the instance -- dataset download and training happen
# entirely on the instance itself, via bootstrap_and_train.sh passed as
# EC2 user-data, so the run can proceed unattended after launch.
#
# Required env vars (no silent defaults for anything security/cost-sensitive):
#   AWS_KEY_NAME           EC2 key pair name (for optional manual SSH access)
#   AWS_SECURITY_GROUP_ID  security group id allowing outbound HTTPS (+ SSH if you want in)
#   AWS_IAM_INSTANCE_PROFILE  instance profile name/ARN with S3 write access to AGENTFORGE_S3_BUCKET
#   AGENTFORGE_S3_BUCKET   s3://bucket/prefix that checkpoints/reports/manifest_stats.json sync to
#   AGENTFORGE_GIT_REMOTE  git URL the instance clones (e.g. https://github.com/Auro-rium/agentforge.git)
#   HF_TOKEN               dual-purpose: gated Salesforce/xlam-function-calling-60k dataset download,
#                           AND publishing the trained adapter to https://huggingface.co/auro-rirum --
#                           needs both dataset-read and repo-write scopes on that account
#
# Optional:
#   AWS_INSTANCE_TYPE      default g5.12xlarge (4x A10G, 24GiB/~22-23GiB usable VRAM EACH -- data-parallel
#                           DDP training, so every GPU independently holds the full quantized model + LoRA
#                           adapters + its own batch's activations, same per-GPU footprint as a single A10G).
#                           On-demand (us-east-1, verified 2026-07): $5.672/hr = $1.418/GPU-hr -- cheaper
#                           per-GPU than every single-GPU option considered (g5.2xlarge ~$1.21/hr,
#                           g6e.xlarge/1x L40S ~$1.861/hr), AND LoRA's small gradient-sync volume (~65M
#                           trainable params, not the full 12B) gives good multi-GPU scaling efficiency --
#                           net effect: roughly the same total training COST as staying single-GPU, but
#                           ~3.2-3.6x faster wall-clock. Real tradeoff versus g6e.xlarge (1x L40S): more
#                           VRAM headroom on the L40S (estimated peak ~15-22GB against ~22-23GB usable on
#                           A10G is tight) vs. meaningfully faster + no costlier on the A10G setup. If it
#                           OOMs: drop training.per_device_train_batch_size to 1 in the config (raise
#                           gradient_accumulation_steps to compensate) or drop training.max_length.
#   AWS_AMI_ID              default: looked up via SSM (latest AWS Deep Learning AMI, PyTorch, Ubuntu 22.04)
#   AWS_REGION              default: aws configure's current region
#   AGENTFORGE_GIT_REF       default: main
#   AGENTFORGE_TRAIN_CONFIG  default: configs/gemma4-12b-qlora.yaml
#   AGENTFORGE_HF_REPO_NAME  default: gemma4-12b-agentforge (published to auro-rirum/<this>)

set -euo pipefail

: "${AWS_KEY_NAME:?Set AWS_KEY_NAME to an existing EC2 key pair name}"
: "${AWS_SECURITY_GROUP_ID:?Set AWS_SECURITY_GROUP_ID to a security group id}"
: "${AWS_IAM_INSTANCE_PROFILE:?Set AWS_IAM_INSTANCE_PROFILE (needs s3:PutObject on AGENTFORGE_S3_BUCKET)}"
: "${AGENTFORGE_S3_BUCKET:?Set AGENTFORGE_S3_BUCKET, e.g. s3://my-bucket/agentforge}"
: "${AGENTFORGE_GIT_REMOTE:?Set AGENTFORGE_GIT_REMOTE to this repos git URL}"
: "${HF_TOKEN:?Set HF_TOKEN (required for the gated Salesforce/xlam-function-calling-60k dataset)}"

AWS_INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-g5.12xlarge}"
AGENTFORGE_GIT_REF="${AGENTFORGE_GIT_REF:-main}"
AGENTFORGE_TRAIN_CONFIG="${AGENTFORGE_TRAIN_CONFIG:-configs/gemma4-12b-qlora.yaml}"

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found on this machine (used only to issue the launch API call, not to run training)" >&2
  exit 1
fi

if [[ -z "${AWS_AMI_ID:-}" ]]; then
  echo "Looking up latest AWS Deep Learning AMI (PyTorch, Ubuntu 22.04, GPU) via SSM..."
  AWS_AMI_ID="$(aws ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/pytorch-2.4-gpu-py311-cu124-ubuntu22.04/latest/ami-id \
    --query 'Parameter.Value' --output text)"
fi
echo "Using AMI: ${AWS_AMI_ID}"

BOOTSTRAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_FILE="$(mktemp)"
trap 'rm -f "${USER_DATA_FILE}"' EXIT

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export AGENTFORGE_S3_BUCKET='${AGENTFORGE_S3_BUCKET}'"
  echo "export AGENTFORGE_GIT_REMOTE='${AGENTFORGE_GIT_REMOTE}'"
  echo "export AGENTFORGE_GIT_REF='${AGENTFORGE_GIT_REF}'"
  echo "export AGENTFORGE_TRAIN_CONFIG='${AGENTFORGE_TRAIN_CONFIG}'"
  echo "export AGENTFORGE_HF_REPO_NAME='${AGENTFORGE_HF_REPO_NAME:-gemma4-12b-agentforge}'"
  echo "export HF_TOKEN='${HF_TOKEN}'"
  cat "${BOOTSTRAP_SCRIPT_DIR}/bootstrap_and_train.sh"
} > "${USER_DATA_FILE}"

# Tries a one-time spot request first (meaningfully cheaper -- see the pricing note below); if
# EC2 can't fulfill it (capacity, price, or AZ constraints -- run-instances fails synchronously
# for a one-time request, no polling needed) falls back to on-demand automatically.
#
# REAL CAVEAT, not yet mitigated: a spot interruption terminates the instance outright (this is a
# one-time request, not persistent), and the current bootstrap_and_train.sh only syncs artifacts
# to S3 at the very end of the whole pipeline, not incrementally during training -- so as things
# stand today, an interruption mid-training loses the whole in-progress run, not just the time
# since the last local checkpoint. Fine for a first/cheap experimental run; if you want real
# interruption resilience (incremental S3 sync of checkpoints + train.py resuming via
# trainer.train(resume_from_checkpoint=...) on relaunch), that's a real follow-up, not implemented
# yet -- ask if you want it built before relying on spot for a long/expensive run.
COMMON_RUN_ARGS=(
  --image-id "${AWS_AMI_ID}"
  --instance-type "${AWS_INSTANCE_TYPE}"
  --key-name "${AWS_KEY_NAME}"
  --security-group-ids "${AWS_SECURITY_GROUP_ID}"
  --iam-instance-profile "Name=${AWS_IAM_INSTANCE_PROFILE}"
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}'
  --user-data "file://${USER_DATA_FILE}"
)

LAUNCH_MODE="spot"
echo "Launching ${AWS_INSTANCE_TYPE} as a spot instance..."
if INSTANCE_ID="$(aws ec2 run-instances \
  "${COMMON_RUN_ARGS[@]}" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=agentforge-train},{Key=agentforge-launch-mode,Value=spot}]' \
  --query 'Instances[0].InstanceId' --output text 2>/tmp/spot_launch_err.txt)"; then
  echo "Spot request fulfilled: ${INSTANCE_ID}"
else
  echo "Spot launch failed (capacity/price/AZ constraint), falling back to on-demand:"
  cat /tmp/spot_launch_err.txt >&2
  LAUNCH_MODE="on-demand"
  INSTANCE_ID="$(aws ec2 run-instances \
    "${COMMON_RUN_ARGS[@]}" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=agentforge-train},{Key=agentforge-launch-mode,Value=on-demand}]' \
    --query 'Instances[0].InstanceId' --output text)"
  echo "On-demand instance launched: ${INSTANCE_ID}"
fi
rm -f /tmp/spot_launch_err.txt

echo "Instance ${INSTANCE_ID} launching. Waiting for it to enter 'running' state..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat <<EOF

Instance ${INSTANCE_ID} is running at ${PUBLIC_IP}.

Data fetch + build + training start automatically via user-data. Progress log:
  ssh -i <your-key>.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/agentforge-bootstrap.log'

Launch mode: ${LAUNCH_MODE}
Artifacts (checkpoints, manifest_stats.json, reports/) sync to ${AGENTFORGE_S3_BUCKET} -- but
only once, at the very end of the whole pipeline, not incrementally during training (see the
spot-interruption caveat above).

Cost: ${AWS_INSTANCE_TYPE} runs ~\$5.672/hr on-demand (us-east-1, verified 2026-07 -- check your
region/current pricing, this drifts). No verified spot figure for this instance type -- if this
launched on spot, check the actual settled price in Billing, don't trust a guessed number here.
Training-time estimate (rough -- real manifest size isn't known until the data pull actually
runs, see technical.md): single-GPU compute would be ~20-30 hours; with 4x A10G data-parallel
DDP and LoRA's small gradient-sync volume giving good scaling efficiency, expect roughly
~6-10 hours wall-clock, i.e. roughly \$34-\$57 for training alone at the on-demand rate, plus a
small amount more for setup/eval/publish. Check manifest_stats.json for the real row/token
counts once the data pull finishes -- that turns this range into a real number, and the
trainer's own steps/sec logging a few minutes into a real run gives a live, accurate ETA.

Remember to terminate the instance when done -- this script does not do that for you:
  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}
EOF
