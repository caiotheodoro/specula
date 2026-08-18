"""Smoke-train settings, unit-testable without Modal."""

from __future__ import annotations

import json

SMOKE_MAX_STEPS = 4


def trainer_kwargs(smoke: bool, epochs: int = 2) -> dict:
    return {
        "load_in_4bit": True,
        "max_steps": SMOKE_MAX_STEPS if smoke else -1,
        "num_train_epochs": 1 if smoke else epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "bf16": True,
        "max_seq_length": 256 if smoke else 1024,
        "output_dir": "/checkpoints/sft",
        # chunked_nll upcasts lm_head to fp32 (~5GB) and is incompatible
        # with PEFT + VLM on L4 24GB.
        "loss_type": "nll",
    }


TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

USER_PROMPT = (
    "Review this food label for FDA compliance. Emit the verdict JSON."
)


def _chat(user: str, assistant: str, image_b64: str | None = None) -> dict:
    if image_b64:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
            ],
            "images": [image_b64],
        }
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def dummy_sft_records() -> list[dict]:
    assistant = (
        '{"verdict":"FLAG","violations":[{"type":"ALLERGEN",'
        '"severity":"CRITICAL","cfr":"FALCPA, 21 CFR 101.4",'
        '"observed":"milk not declared","expected":"milk must be declared",'
        '"correction":"declare milk in ingredients or Contains"}]}'
    )
    return [_chat(USER_PROMPT, assistant, TINY_PNG_B64)]


def sft_records_from_bytes(data: bytes) -> list[dict]:
    """Parse Modal --data payload: empty, TRL messages, builder shards, or Task JSONL."""
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()] if data else []
    if not lines:
        return dummy_sft_records()
    records = []
    for ln in lines:
        obj = json.loads(ln)
        if "messages" in obj:
            rec = {"messages": obj["messages"]}
            if "images" in obj:
                rec["images"] = obj["images"]
            elif obj.get("image_b64"):
                rec["images"] = [obj["image_b64"]]
            records.append(rec)
            continue
        if "assistant_json" in obj:
            records.append(_chat(
                obj.get("user_text") or USER_PROMPT,
                obj["assistant_json"],
                obj.get("image_b64"),
            ))
            continue
        if "expected" in obj:
            expected = obj["expected"]
            assistant = (
                expected if isinstance(expected, str)
                else json.dumps(expected, separators=(",", ":"))
            )
            records.append(_chat(USER_PROMPT, assistant, obj.get("image_b64")))
            continue
        raise ValueError(f"unrecognized SFT record keys: {sorted(obj)}")
    return records
