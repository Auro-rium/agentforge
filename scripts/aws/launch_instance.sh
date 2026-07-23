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
#   HF_TOKEN               needed on-instance for the gated Salesforce/xlam-function-calling-60k dataset
#
# Optional:
#   AWS_INSTANCE_TYPE      default g5.2xlarge (1x A10G, 24GB -- fits gemma-4-12B-it QLoRA)
#   AWS_AMI_ID              default: looked up via SSM (latest AWS Deep Learning AMI, PyTorch, Ubuntu 22.04)
#   AWS_REGION              default: aws configure's current region
#   AGENTFORGE_GIT_REF       default: main
#   AGENTFORGE_TRAIN_CONFIG  default: configs/gemma4-12b-qlora.yaml

set -euo pipefail

: "${AWS_KEY_NAME:?Set AWS_KEY_NAME to an existing EC2 key pair name}"
: "${AWS_SECURITY_GROUP_ID:?Set AWS_SECURITY_GROUP_ID to a security group id}"
: "${AWS_IAM_INSTANCE_PROFILE:?Set AWS_IAM_INSTANCE_PROFILE (needs s3:PutObject on AGENTFORGE_S3_BUCKET)}"
: "${AGENTFORGE_S3_BUCKET:?Set AGENTFORGE_S3_BUCKET, e.g. s3://my-bucket/agentforge}"
: "${AGENTFORGE_GIT_REMOTE:?Set AGENTFORGE_GIT_REMOTE to this repo's git URL}"
: "${HF_TOKEN:?Set HF_TOKEN (required for the gated Salesforce/xlam-function-calling-60k dataset)}"

AWS_INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-g5.2xlarge}"
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
  echo "export HF_TOKEN='${HF_TOKEN}'"
  cat "${BOOTSTRAP_SCRIPT_DIR}/bootstrap_and_train.sh"
} > "${USER_DATA_FILE}"

echo "Launching ${AWS_INSTANCE_TYPE}..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "${AWS_AMI_ID}" \
  --instance-type "${AWS_INSTANCE_TYPE}" \
  --key-name "${AWS_KEY_NAME}" \
  --security-group-ids "${AWS_SECURITY_GROUP_ID}" \
  --iam-instance-profile "Name=${AWS_IAM_INSTANCE_PROFILE}" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
  --user-data "file://${USER_DATA_FILE}" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=agentforge-train}]' \
  --query 'Instances[0].InstanceId' --output text)"

echo "Instance ${INSTANCE_ID} launching. Waiting for it to enter 'running' state..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat <<EOF

Instance ${INSTANCE_ID} is running at ${PUBLIC_IP}.

Data fetch + build + training start automatically via user-data. Progress log:
  ssh -i <your-key>.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/agentforge-bootstrap.log'

Artifacts (checkpoints, manifest_stats.json, reports/) sync to:
  ${AGENTFORGE_S3_BUCKET}

Remember to terminate the instance when done -- this script does not do that for you:
  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}
EOF
