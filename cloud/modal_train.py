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

import sys
from pathlib import Path

import modal

_MODEL_SRC = Path(__file__).resolve().parent.parent / "model" / "src"
if str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))

from specula_model.train_config import dummy_sft_records, trainer_kwargs

app = modal.App("specula-train")
vol = modal.Volume.from_name("specula-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile").add_local_python_source(
    "specula_model"
)

GPU_L4 = modal.gpu.L4(count=1)


@app.function(image=image, gpu=GPU_L4, volumes={"/checkpoints": vol},
              timeout=60 * 60 * 6)
def train(data: bytes, smoke: bool = False, epochs: int = 2) -> str:
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    kw = trainer_kwargs(smoke=smoke, epochs=epochs)
    # Wave 0a: dummy records only; JSONL parse of `data` is a later slice.
    records = dummy_sft_records()
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", torch_dtype="bfloat16",
        load_in_4bit=kw["load_in_4bit"])
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    cfg = SFTConfig(
        output_dir=kw["output_dir"], max_seq_length=kw["max_seq_length"],
        per_device_train_batch_size=kw["per_device_train_batch_size"],
        gradient_accumulation_steps=kw["gradient_accumulation_steps"],
        gradient_checkpointing=kw["gradient_checkpointing"], bf16=kw["bf16"],
        logging_steps=10, num_train_epochs=kw["num_train_epochs"],
        max_steps=kw["max_steps"],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=records)
    trainer.train()
    trainer.save_model("/checkpoints/sft-final")
    vol.commit()
    return "done"


@app.local_entrypoint()
def main(smoke: bool = False, epochs: int = 2, data: str = "") -> None:
    blob = Path(data).read_bytes() if data else b""
    train.remote(blob, smoke=smoke, epochs=epochs)
