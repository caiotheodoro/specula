#!/usr/bin/env bash
# On-demand g6.2xlarge (1x L4) GRPO. Mirrors cloud/gcp_spot.sh.
# GPU quota must be >0 (G/VT On-Demand, L-DB2E81BA). Checkpoints land in S3.
#
#   ./cloud/aws_grpo.sh
#   SMOKE=1 ./cloud/aws_grpo.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=$REGION
BUCKET=${AWS_BUCKET:-specula-grpo-${ACCOUNT}}
PREFIX=${AWS_PREFIX:-grpo}
NAME=${AWS_NAME:-specula-rlvr-$(date +%m%d-%H%M)}
ITYPE=${AWS_INSTANCE:-g6.2xlarge}
SMOKE=${SMOKE:-0}
ITERS=${ITERS:-1200}
GROUP_SIZE=${GROUP_SIZE:-2}
ADAPTER_SRC=${ADAPTER_SRC:-/tmp/specula-aws/ckpts/sft-7b}
ROLE_NAME=specula-grpo-ec2
SG_NAME=specula-grpo-ssm
AMI_NAME_GLOB='Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.*Ubuntu 22.04*'

quota=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query Quota.Value --output text)
if [ "${quota%.*}" -lt 8 ]; then
  echo "G/VT On-Demand quota is ${quota} vCPU in ${REGION} (need 8 for ${ITYPE})." >&2
  echo "Pending: aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region ${REGION}" >&2
  exit 2
fi

if [ ! -f "${ADAPTER_SRC}/adapter_model.safetensors" ]; then
  echo "SFT adapter missing at ${ADAPTER_SRC}" >&2
  exit 1
fi
if [ ! -s "${ROOT}/forge/data/rlvr-prompts.jsonl" ]; then
  echo "missing ${ROOT}/forge/data/rlvr-prompts.jsonl" >&2
  exit 1
fi

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name specula-s3 --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\",\"s3:DeleteObject\"],
      \"Resource\":[\"arn:aws:s3:::${BUCKET}\",\"arn:aws:s3:::${BUCKET}/*\"]}]
  }"
fi
if ! aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE_NAME"
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
  sleep 12
fi

VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
  SG=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "specula GRPO SSM-only" \
    --vpc-id "$VPC" --query GroupId --output text)
fi

AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=${AMI_NAME_GLOB}" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
SUBNET=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text)

TAR=/tmp/specula-src.tgz
tar -C "$ROOT" --exclude .venv --exclude .git --exclude __pycache__ \
  --exclude '*.pyc' --exclude .pytest_cache --exclude .env --exclude '.env.*' \
  --exclude '*.jsonl' --exclude forge/data \
  -czf "$TAR" model/src forge/src cloud/aws_grpo_start.sh
aws s3 cp "$TAR" "s3://${BUCKET}/${PREFIX}/specula-src.tgz"
aws s3 cp "${ROOT}/forge/data/rlvr-prompts.jsonl" "s3://${BUCKET}/${PREFIX}/rlvr-prompts.jsonl"
aws s3 sync "$ADAPTER_SRC" "s3://${BUCKET}/${PREFIX}/sft-7b/"
aws s3 cp "${ROOT}/cloud/aws_grpo_start.sh" "s3://${BUCKET}/${PREFIX}/aws_grpo_start.sh"

ITERS_EFFECTIVE=$ITERS
if [ "$SMOKE" = "1" ]; then ITERS_EFFECTIVE=2; fi

UD=$(cat <<EOF
#!/bin/bash
set -euxo pipefail
export AWS_DEFAULT_REGION=${REGION}
export BUCKET=${BUCKET}
export PREFIX=${PREFIX}
export ITERS=${ITERS_EFFECTIVE}
export GROUP_SIZE=${GROUP_SIZE}
aws s3 cp s3://${BUCKET}/${PREFIX}/aws_grpo_start.sh /tmp/aws_grpo_start.sh
chmod +x /tmp/aws_grpo_start.sh
bash /tmp/aws_grpo_start.sh
EOF
)

IID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$ITYPE" \
  --subnet-id "$SUBNET" \
  --security-group-ids "$SG" \
  --iam-instance-profile Name="$ROLE_NAME" \
  --associate-public-ip-address \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=project,Value=specula}]" \
  --user-data "$UD" \
  --query 'Instances[0].InstanceId' --output text)

echo "started ${IID} (${ITYPE} ${REGION})  s3://${BUCKET}/${PREFIX}/"
echo "logs: aws s3 cp s3://${BUCKET}/${PREFIX}/specula-grpo.log -"
echo "ssm:  aws ssm start-session --target ${IID} --region ${REGION}"
echo "$IID"
