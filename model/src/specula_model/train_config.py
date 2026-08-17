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


def dummy_sft_records() -> list[dict]:
    # TRL 1.x conversational LM format (not user_text/assistant_json).
    assistant = (
        '{"verdict":"FLAG","violations":[{"type":"ALLERGEN",'
        '"severity":"CRITICAL","cfr":"FALCPA, 21 CFR 101.4",'
        '"observed":"milk not declared","expected":"milk must be declared",'
        '"correction":"declare milk in ingredients or Contains"}]}'
    )
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Review this food label for FDA compliance. "
                        "Emit the verdict JSON."
                    ),
                },
                {"role": "assistant", "content": assistant},
            ],
        }
    ]


USER_PROMPT = (
    "Review this food label for FDA compliance. Emit the verdict JSON."
)


def _chat(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def sft_records_from_bytes(data: bytes) -> list[dict]:
    """Parse Modal --data payload: empty, TRL messages, builder shards, or Task JSONL."""
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()] if data else []
    if not lines:
        return dummy_sft_records()
    records = []
    for ln in lines:
        obj = json.loads(ln)
        if "messages" in obj:
            records.append({"messages": obj["messages"]})
            continue
        if "assistant_json" in obj:
            records.append(_chat(obj.get("user_text") or USER_PROMPT, obj["assistant_json"]))
            continue
        if "expected" in obj:
            expected = obj["expected"]
            assistant = (
                expected if isinstance(expected, str)
                else json.dumps(expected, separators=(",", ":"))
            )
            records.append(_chat(USER_PROMPT, assistant))
            continue
        raise ValueError(f"unrecognized SFT record keys: {sorted(obj)}")
    return records
