"""Modal training app (primary cloud, free $30/mo credit, L4 24GB).

QLoRA SFT on Qwen/Qwen2.5-VL-7B-Instruct. Smoke first (--smoke) to validate DeltaNet
tooling, then full runs. Checkpoints persist to a Modal Volume so runs resume
across monthly credit cycles. GPU: L4 (24GB) fits 4-bit QLoRA of the 28B VL
model with gradient checkpointing.

Run:
  modal run cloud/modal_train.py --smoke
  modal run cloud/modal_train.py --epochs 3 --data data/train.jsonl
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("specula-train")
vol = modal.Volume.from_name("specula-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("specula-hf-cache", create_if_missing=True)
_DOCKER = Path(__file__).resolve().parent / "Dockerfile"
image = modal.Image.from_dockerfile(str(_DOCKER)).add_local_python_source(
    "specula_model"
)



def _with_pil_images(records: list[dict]) -> list[dict]:
    """Decode image_b64 / images strings to RGB PIL, thumbnail for L4 VRAM."""
    import base64
    from io import BytesIO
    from PIL import Image

    out = []
    for rec in records:
        rec = dict(rec)
        items = rec.get("images") or []
        if not items:
            out.append(rec)
            continue
        pil = []
        for item in items:
            if isinstance(item, str):
                img = Image.open(BytesIO(base64.b64decode(item))).convert("RGB")
            else:
                img = item
            img.thumbnail((384, 384))
            pil.append(img)
        rec["images"] = pil
        out.append(rec)
    return out


@app.function(
    image=image,
    gpu="L4",
    volumes={"/checkpoints": vol, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 60 * 6,
)
def train(data: bytes, smoke: bool = False, epochs: int = 2,
          resume: str = "", out: str = "") -> str:
    import sys
    from datasets import Dataset
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from specula_model.train_config import (
        load_base_vl, sft_records_from_bytes, sft_save_paths, trainer_kwargs)

    resume, save_dir = sft_save_paths(resume=resume, out=out)
    kw = trainer_kwargs(smoke=smoke, epochs=epochs)
    records = _with_pil_images(sft_records_from_bytes(data))
    if smoke and len(records) < 16:
        records = (records * 16)[:16]
    bnb = BitsAndBytesConfig(
        load_in_4bit=kw["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model, processor = load_base_vl(bnb)
    model.config.use_cache = False
    adapter_path = Path(resume) if resume else None
    if adapter_path is not None and adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        if resume and not smoke:
            raise FileNotFoundError(f"SFT adapter missing: {resume}")
        lora = LoraConfig(r=8 if smoke else 32, lora_alpha=16 if smoke else 64,
                          lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    cfg = SFTConfig(
        output_dir=kw["output_dir"], max_length=kw["max_seq_length"],
        per_device_train_batch_size=kw["per_device_train_batch_size"],
        gradient_accumulation_steps=kw["gradient_accumulation_steps"],
        gradient_checkpointing=kw["gradient_checkpointing"], bf16=kw["bf16"],
        logging_steps=1, num_train_epochs=kw["num_train_epochs"],
        max_steps=kw["max_steps"], report_to="none",
        loss_type=kw["loss_type"],
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=Dataset.from_list(records),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(save_dir)
    vol.commit()
    hf_cache.commit()
    return save_dir


@app.local_entrypoint()
def main(smoke: bool = False, epochs: int = 2, data: str = "",
         resume: str = "", out: str = "") -> None:
    blob = Path(data).read_bytes() if data else b""
    if smoke:
        print(train.remote(blob, smoke=smoke, epochs=epochs, resume=resume, out=out))
        return
    call = train.spawn(blob, smoke=smoke, epochs=epochs, resume=resume, out=out)
    print(call.object_id)
