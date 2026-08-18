"""Smoke-train settings, unit-testable without Modal."""

from __future__ import annotations

import json

from .schema import SYSTEM_PROMPT

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



RLVR_SMOKE_MAX_STEPS = 2


def grpo_kwargs(smoke: bool, iters: int = 200, group_size: int = 8) -> dict:
    """Dr. GRPO + DAPO decoupled clip. Batch size must divide by G."""
    g = 2 if smoke else group_size
    return {
        "load_in_4bit": True,
        "loss_type": "dr_grpo",
        "epsilon": 0.2,
        "epsilon_high": 1.0,
        "num_generations": g,
        "per_device_train_batch_size": g,
        "gradient_accumulation_steps": 1 if smoke else 4,
        "gradient_checkpointing": True,
        "bf16": True,
        "max_steps": RLVR_SMOKE_MAX_STEPS if smoke else iters,
        "max_completion_length": 64 if smoke else 256,
        "max_prompt_length": 256 if smoke else 512,
        "output_dir": "/checkpoints/rlvr",
        "scale_rewards": False,
        "remove_unused_columns": False,
    }


def dummy_rlvr_records() -> list[dict]:
    """Smoke prompts: image + system/user, no assistant gold."""
    rec = {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "images": [TINY_PNG_B64],
        "expected_verdict": '{"verdict":"PASS","violations":[]}',
    }
    return [{**rec, "task_id": f"smoke-{i}"} for i in range(4)]


def rlvr_records_from_bytes(data: bytes) -> list[dict]:
    """Parse RLVR prompt JSONL. Empty -> dummy. SFT traces are rejected."""
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()] if data else []
    if not lines:
        return dummy_rlvr_records()
    records = []
    for ln in lines:
        obj = json.loads(ln)
        if "assistant_json" in obj or obj.get("assistant"):
            raise ValueError("SFT traces cannot be used as RLVR prompts")
        messages = obj.get("messages") or obj.get("prompt") or []
        if any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
            raise ValueError("SFT traces cannot be used as RLVR prompts")
        if "prompt" not in obj or "expected_verdict" not in obj:
            raise ValueError(f"unrecognized RLVR record keys: {sorted(obj)}")
        expected = obj["expected_verdict"]
        if not isinstance(expected, str):
            expected = json.dumps(expected, separators=(",", ":"))
        images = obj.get("images") or []
        if not images and obj.get("image_b64"):
            images = [obj["image_b64"]]
        records.append({
            "task_id": obj.get("task_id", ""),
            "prompt": obj["prompt"],
            "images": images,
            "expected_verdict": expected,
        })
    return records



def grpo_trainer_kwargs(kw: dict, accepted: set[str]) -> dict:
    """Keep GRPOConfig fields that this TRL version actually accepts.

    `load_in_4bit` is a load flag, not a GRPOConfig field. Newer TRL dropped
    `max_prompt_length`; filter so Modal's unpinned `trl>=0.16` does not crash.
    """
    skip = {"load_in_4bit"}
    return {k: v for k, v in kw.items() if k not in skip and k in accepted}
