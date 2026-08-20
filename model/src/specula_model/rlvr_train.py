"""GRPO RLVR loop shared by Modal and GCP.

Loads the SFT adapter, samples G completions per prompt, scores them with
the forge oracle, and updates with Dr. GRPO + DAPO decoupled clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .train_config import grpo_kwargs, grpo_trainer_kwargs, load_base_vl, rlvr_records_from_bytes, rlvr_save_dir


def _with_pil_images(records: list[dict]) -> list[dict]:
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
            img.thumbnail((256, 256))
            pil.append(img)
        rec["images"] = pil
        out.append(rec)
    return out


def run_grpo(data: bytes, smoke: bool = False, adapter: str = "/checkpoints/sft-7b",
             iters: int = 200, group_size: int = 2,
             checkpoint_dir: str = "/checkpoints",
             save_name: str = "rlvr-final") -> str:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from .rlvr_reward import oracle_reward_func

    kw = grpo_kwargs(smoke=smoke, iters=iters, group_size=group_size)
    records = _with_pil_images(rlvr_records_from_bytes(data))
    if smoke and len(records) < 4:
        records = (records * 4)[:4]
    out_dir = str(Path(checkpoint_dir) / "rlvr")
    final_dir = rlvr_save_dir(checkpoint_dir, save_name)
    bnb = BitsAndBytesConfig(
        load_in_4bit=kw["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model, processor = load_base_vl(bnb)
    adapter_path = Path(adapter)
    if adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        if not smoke:
            raise FileNotFoundError(f"SFT adapter missing: {adapter_path}")
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    tok = getattr(processor, "tokenizer", processor)
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None) or eos_id
    for obj in (processor, tok):
        if hasattr(obj, "padding_side"):
            try:
                obj.padding_side = "left"
            except (AttributeError, TypeError):
                pass
    # TRL GRPOTrainer reads tokenizer fields on processing_class. Qwen2.5-VL
    # processors keep those on the nested tokenizer.
    for attr in (
        "bos_token", "bos_token_id", "eos_token", "eos_token_id",
        "pad_token", "pad_token_id", "unk_token", "unk_token_id",
        "padding_side",
    ):
        if not hasattr(processor, attr):
            try:
                setattr(processor, attr, getattr(tok, attr, None))
            except (AttributeError, TypeError):
                pass
    if hasattr(model, "generation_config"):
        if eos_id is not None:
            model.generation_config.eos_token_id = eos_id
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id
    import dataclasses
    accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    cfg_kw = grpo_trainer_kwargs(kw, accepted)
    extra = {
        "output_dir": out_dir,
        "logging_steps": 1,
        "report_to": "none",
        "temperature": 0.7,
        "save_steps": max(kw["max_steps"], 1) if smoke else 20,
        "generation_kwargs": {"eos_token_id": eos_id, "pad_token_id": pad_id},
    }
    cfg_kw.update({k: v for k, v in extra.items() if k in accepted or k == "output_dir"})
    cfg = GRPOConfig(**{k: v for k, v in cfg_kw.items() if k in accepted or k == "output_dir"})
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=oracle_reward_func,
        processing_class=processor,
        args=cfg,
        train_dataset=Dataset.from_list(records),
    )
    trainer.train()
    trainer.save_model(final_dir)
    return final_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="",
                    help="RLVR prompt JSONL (not SFT traces)")
    ap.add_argument("--adapter", default="/checkpoints/sft-7b")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoint-dir", default="/checkpoints")
    ap.add_argument("--save-name", default="rlvr-final")
    args = ap.parse_args()
    blob = Path(args.prompts).read_bytes() if args.prompts else b""
    print(run_grpo(blob, smoke=args.smoke, adapter=args.adapter,
                   iters=args.iters, group_size=args.group_size,
                   checkpoint_dir=args.checkpoint_dir,
                   save_name=args.save_name))


if __name__ == "__main__":
    main()
