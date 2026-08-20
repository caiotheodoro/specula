#!/usr/bin/env bash
# SageMaker training entry. No shutdown — SM packs /opt/ml/model.
set -euxo pipefail
export BUCKET=${BUCKET:?set BUCKET}
export PREFIX=${PREFIX:-grpo}
export ITERS=${ITERS:-1200}
export GROUP_SIZE=${GROUP_SIZE:-2}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
export HF_HOME=/opt/ml/hf-cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/specula/model/src:/opt/specula/forge/src
mkdir -p /opt/specula /opt/specula/forge/data /opt/specula-ckpts "$HF_HOME" /opt/ml/model

python3 - <<'PY'
import os, boto3
b = os.environ["BUCKET"]
p = os.environ["PREFIX"]
s3 = boto3.client("s3")
s3.download_file(b, f"{p}/specula-src.tgz", "/tmp/specula-src.tgz")
s3.download_file(b, f"{p}/rlvr-prompts.jsonl", "/opt/specula/forge/data/rlvr-prompts.jsonl")
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=b, Prefix=f"{p}/sft-7b/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        dest = "/opt/specula-ckpts/" + key[len(p)+1:]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        s3.download_file(b, key, dest)
print("s3 download ok")
PY

tar -C /opt/specula -xzf /tmp/specula-src.tgz
python3 -m pip install --upgrade pip
# Image has torch 2.5.1; trl 1.x and transformers 5.x need newer torch.
python3 -m pip install "transformers>=4.55,<5" "trl>=0.16,<0.18" \
  "peft>=0.15,<0.18" "accelerate>=1.0,<1.6"
python3 -c "import torch; assert torch.cuda.is_available(), torch.__version__; print(torch.__version__, torch.cuda.get_device_name(0))"

cd /opt/specula
python3 -m specula_model.rlvr_train \
  --prompts /opt/specula/forge/data/rlvr-prompts.jsonl \
  --adapter /opt/specula-ckpts/sft-7b \
  --iters "${ITERS}" --group-size "${GROUP_SIZE}" \
  --checkpoint-dir /opt/ml/model \
  --save-name rlvr-7b

python3 - <<'PY'
import os, boto3
from pathlib import Path
b = os.environ["BUCKET"]
p = os.environ["PREFIX"]
s3 = boto3.client("s3")
root = Path("/opt/ml/model")
for f in root.rglob("*"):
    if f.is_file():
        key = f"{p}/sagemaker-out/{f.relative_to(root)}"
        s3.upload_file(str(f), b, key)
        print("uploaded", key)
PY
