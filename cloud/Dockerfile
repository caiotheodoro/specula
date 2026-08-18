FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip python3-venv curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Specula training deps (SOTA 2026 stack)
RUN pip install --no-cache-dir \
    "torch>=2.6" "transformers>=4.55" "peft>=0.13" "trl>=0.16" \
    "accelerate>=1.0" "datasets>=3.0" "pillow>=11" "huggingface_hub>=0.26" \
    "bitsandbytes>=0.44"

# Modal inspects `python` on PATH; Ubuntu 22.04 only ships python3/pip3.
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Qwen3 VL AutoProcessor imports Qwen3VLVideoProcessor → needs torchvision.
RUN pip install --no-cache-dir torchvision

# Triton JIT-compiles cuda_utils.c on first forward; needs Python.h.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pydantic>=2.7"

COPY . .

CMD ["python3", "-c", "print('specula-train-image-ok')"]