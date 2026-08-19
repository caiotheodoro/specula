"""Adapter eval: RLVR-shaped prompts, oracle scoring, OpenAI-compat helpers.

GPU generate lives here so Modal and a local server share one path.
CPU tests cover scoring, resume, and request parsing — not weight load.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from .schema import SYSTEM_PROMPT, parse, to_forge_verdict
from .train_config import USER_PROMPT

EVAL_THUMB = (256, 256)
EVAL_MAX_NEW_TOKENS = 256
MODEL_ID = "Qwen/Qwen3.8-27B"


def eval_messages(image=None) -> list[dict]:
    """Same system+user shape as RLVR prompts (no assistant gold)."""
    image_part: dict = {"type": "image"}
    if image is not None:
        image_part = {"type": "image", "image": image}
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            image_part,
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]


def thumbnail_png(png: bytes):
    from PIL import Image
    img = Image.open(BytesIO(png)).convert("RGB")
    img.thumbnail(EVAL_THUMB)
    return img


def score_raw_completion(task, raw: str) -> dict:
    from specula_forge.score import score_predictions

    predicted = to_forge_verdict(raw)
    out = parse(raw)
    return {
        "task_id": task.task_id,
        "expected": task.expected.model_dump(),
        "predicted": out.model_dump() if out else None,
        "raw": raw,
        "score": score_predictions(task.expected, predicted),
    }


def run_eval(tasks_jsonl: Path, generate_fn, out: Path, limit: int | None = None) -> dict:
    from specula_forge.generate import render_png
    from specula_forge.schema import Task
    from specula_forge.score import summarize

    tasks = [Task.model_validate_json(l) for l in
             tasks_jsonl.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["task_id"])
    remaining = [t for t in tasks if t.task_id not in done_ids]
    if limit is not None:
        remaining = remaining[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for t in remaining:
            png = render_png(t.label)
            raw = generate_fn(t, png)
            row = score_raw_completion(t, raw)
            f.write(json.dumps(row) + "\n")
            f.flush()
    scores = []
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "score" in rec:
            scores.append(rec["score"])
    return summarize(scores)


def openai_compat_reply(model: str, content: str) -> dict:
    return {
        "id": "specula-eval",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


def parse_chat_request(body: dict) -> dict:
    image = b""
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = ""
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url") or ""
            if url.startswith("data:") and "," in url:
                image = base64.b64decode(url.split(",", 1)[1])
    return {"model": body.get("model") or "specula", "image": image}


def benchmark_row(model: str, summary: dict) -> dict:
    return {
        "model": model,
        "severity_weighted_recall": summary.get("severity_weighted_recall"),
        "critical_recall": summary.get("critical_recall"),
        "high_recall": summary.get("high_recall"),
        "precision": summary.get("precision"),
        "parse_rate": summary.get("parse_rate"),
        "n_tasks": summary.get("n_tasks"),
    }


def load_vl_adapter(adapter: str):
    """4-bit Qwen3.8-27B + optional PEFT adapter. GPU only."""
    from pathlib import Path as _Path
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype="bfloat16", quantization_config=bnb)
    if adapter and _Path(adapter).exists():
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = getattr(processor, "tokenizer", processor)
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None) or eos_id
    if hasattr(model, "generation_config"):
        if eos_id is not None:
            model.generation_config.eos_token_id = eos_id
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id
    return model, processor


def generate_from_png(model, processor, png: bytes,
                      max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> str:
    import torch
    pil = thumbnail_png(png)
    messages = eval_messages()
    try:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    inputs = processor(text=[text], images=[pil], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    moved = {}
    for k, v in inputs.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    prompt_len = moved["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = out[0][prompt_len:]
    tok = getattr(processor, "tokenizer", processor)
    decode = getattr(processor, "decode", None) or tok.decode
    return decode(new_tokens, skip_special_tokens=True)
