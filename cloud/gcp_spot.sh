#!/usr/bin/env bash
# GCP $300 fallback: spot L4/A100 training marathon under strict budget.
# Prereq: gcloud active account you@example.com, project
# your-gcp-project, Compute API enabled (done 2026-08-17).
set -euo pipefail

PROJECT=${GCP_PROJECT:?set GCP_PROJECT}
ZONE=${GCP_ZONE:-us-east1-b}
MACHINE=${GCP_MACHINE:-n1-standard-4}   # 1x L4
BOOT=projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts
NAME="plumb-train-$(date +%m%d-%H%M)"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --preemptible \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --metadata=startup-script='#!/bin/bash
apt-get update && apt-get install -y python3-pip
pip3 install torch transformers peft trl accelerate datasets bitsandbytes flash-attn unsloth vllm
cd /root && git clone https://github.com/caiotheodoro/plumb.git && cd plumb
python3 -m plumb_model.train --data data/train.jsonl 2>&1 | tee /root/train.log'

echo "started $NAME (spot L4) — gcloud compute ssh $NAME --zone=$ZONE"