#!/usr/bin/env bash
# Runs on the GPU instance. Pulled from S3 by user-data.
set -euxo pipefail
exec > >(tee -a /var/log/specula-grpo.log) 2>&1

BUCKET=${BUCKET:?}
PREFIX=${PREFIX:-grpo}
ITERS=${ITERS:-1200}
GROUP_SIZE=${GROUP_SIZE:-2}

export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
export HF_HOME=/opt/hf-cache
export PYTHONUNBUFFERED=1
mkdir -p /opt/specula /opt/specula-ckpts /opt/specula/forge/data "$HF_HOME"

sync_logs() {
  aws s3 cp /var/log/specula-grpo.log "s3://${BUCKET}/${PREFIX}/specula-grpo.log" || true
}
(while true; do sleep 120; sync_logs; done) &

for i in $(seq 1 60); do
  if command -v nvidia-smi >/dev/null && nvidia-smi; then
    break
  fi
  sleep 10
done
nvidia-smi

if [ -f /opt/pytorch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch || true
fi

aws s3 cp "s3://${BUCKET}/${PREFIX}/specula-src.tgz" /tmp/specula-src.tgz
tar -C /opt/specula -xzf /tmp/specula-src.tgz
aws s3 cp "s3://${BUCKET}/${PREFIX}/rlvr-prompts.jsonl" /opt/specula/forge/data/rlvr-prompts.jsonl
aws s3 sync "s3://${BUCKET}/${PREFIX}/sft-7b/" /opt/specula-ckpts/sft-7b/

python3 -m pip install --upgrade pip
python3 -m pip install "transformers>=4.55" "peft>=0.13" "trl>=0.16" \
  "accelerate>=1.0" "datasets>=3.0" "pillow>=11" "huggingface_hub>=0.26" \
  "bitsandbytes>=0.44" "pydantic>=2.7"
python3 -c "import torch; assert torch.cuda.is_available(), torch.__version__; print(torch.__version__, torch.cuda.get_device_name(0))"

export PYTHONPATH=/opt/specula/model/src:/opt/specula/forge/src
cd /opt/specula
python3 -m specula_model.rlvr_train \
  --prompts /opt/specula/forge/data/rlvr-prompts.jsonl \
  --adapter /opt/specula-ckpts/sft-7b \
  --iters "${ITERS}" --group-size "${GROUP_SIZE}" \
  --checkpoint-dir /opt/specula-ckpts \
  --save-name rlvr-7b

aws s3 sync /opt/specula-ckpts/rlvr-7b/ "s3://${BUCKET}/${PREFIX}/rlvr-7b/"
aws s3 sync /opt/specula-ckpts/rlvr/ "s3://${BUCKET}/${PREFIX}/rlvr-steps/" || true
echo DONE > /tmp/specula-done
aws s3 cp /tmp/specula-done "s3://${BUCKET}/${PREFIX}/DONE"
sync_logs
shutdown -h now
