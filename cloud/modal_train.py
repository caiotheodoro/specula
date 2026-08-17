"""Modal training app (primary cloud, free $30/mo credit, L4 24GB).

QLoRA SFT on Qwen/Qwen3.8-27B. Smoke first (--smoke) to validate DeltaNet
tooling, then full runs. Checkpoints persist to a Modal Volume so runs resume
across monthly credit cycles. GPU: L4 (24GB) fits 4-bit QLoRA of the 28B VL
model with gradient checkpointing.

Run:
  modal run cloud/modal_train.py --smoke
  modal run cloud/modal_train.py --epochs 3 --data data/train.jsonl
"""

from __future__ import annotations

import modal

app = modal.App("labelforge-train")
vol = modal.Volume.from_name("labelforge-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile")

GPU_L4 = modal.gpu.L4(count=1)


@app.function(image=image, gpu=GPU_L4, volumes={"/checkpoints": vol},
              timeout=60 * 60 * 6)
def train(data: bytes, smoke: bool = False, epochs: int = 2) -> str:
    import io
    import json

    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # TODO(P3): load dataset (image b64 + assistant JSON) from `data` blob,
    # build Qwen3.8 chat template, pack sequences, loss-mask padding.
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", torch_dtype="bfloat16",
        load_in_4bit=not smoke)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    cfg = SFTConfig(
        output_dir="/checkpoints/sft", max_seq_length=4096,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, bf16=True, logging_steps=10,
        num_train_epochs=1 if smoke else epochs,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=[])
    trainer.train()
    trainer.save_model("/checkpoints/sft-final")
    vol.commit()
    return "done"