#!/usr/bin/env bash
# GCP $300 marathon: spot L4 GRPO. Modal is for short iteration.
# Does not clone GitHub (this branch may be unpushed). Uploads this checkout.
#
#   ./cloud/gcp_spot.sh              # 2-step smoke on the VM
#   SMOKE=0 GROUP_SIZE=4 ITERS=200 ./cloud/gcp_spot.sh
#
# BLOCKED 2026-08-18: project your-gcp-project has GPUS_ALL_REGIONS=0
# (regional NVIDIA_L4_GPUS=1 is not enough). Instance create fails with
# Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
# Request a GPUS_ALL_REGIONS increase before this script can launch an L4.
#
# Prereq: gcloud account you@example.com, project
# your-gcp-project, Compute API enabled. Copy /checkpoints/sft-final
# onto the VM (Modal volume) before a full run; smoke can start a fresh LoRA.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=${GCP_PROJECT:?set GCP_PROJECT}
ZONE=${GCP_ZONE:-us-east1-b}
MACHINE=${GCP_MACHINE:-n1-standard-4}
NAME=${GCP_NAME:-specula-rlvr-$(date +%m%d-%H%M)}
SMOKE=${SMOKE:-1}
ITERS=${ITERS:-200}
GROUP_SIZE=${GROUP_SIZE:-2}
PROMPTS=${PROMPTS:-}
ADAPTER=${ADAPTER:-/opt/specula-ckpts/sft-final}

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --preemptible \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --metadata=install-nvidia-driver=True

echo "waiting for SSH on $NAME"
for i in $(seq 1 36); do
  if gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command="true" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

TAR=/tmp/specula-src.tgz
tar -C "$ROOT" --exclude .venv --exclude .git --exclude __pycache__ \
    --exclude '*.pyc' --exclude .pytest_cache -czf "$TAR" .
gcloud compute scp --project="$PROJECT" --zone="$ZONE" "$TAR" "$NAME":/tmp/specula-src.tgz

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
sudo mkdir -p /opt/specula /opt/specula-ckpts
sudo tar -C /opt/specula -xzf /tmp/specula-src.tgz
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv build-essential
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install transformers peft trl accelerate datasets bitsandbytes pillow pydantic
export PYTHONPATH=/opt/specula/model/src:/opt/specula/forge/src
cd /opt/specula
EXTRA=""
if [ "${SMOKE}" = "1" ]; then EXTRA="--smoke"; fi
if [ -n "${PROMPTS}" ]; then EXTRA="\$EXTRA --prompts ${PROMPTS}"; fi
python3 -m specula_model.rlvr_train \$EXTRA --adapter ${ADAPTER} --iters ${ITERS} --group-size ${GROUP_SIZE} --checkpoint-dir /opt/specula-ckpts
EOF
)

gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command="$REMOTE_CMD"
echo "started $NAME (spot L4) — gcloud compute ssh $NAME --zone=$ZONE"
