"""Smoke-train settings, unit-testable without Modal."""

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
        "max_seq_length": 4096,
        "output_dir": "/checkpoints/sft",
    }


def dummy_sft_records() -> list[dict]:
    return [
        {
            "user_text": (
                "Review this food label for FDA compliance. Emit the verdict JSON."
            ),
            "assistant_json": (
                '{"verdict":"FLAG","violations":[{"type":"ALLERGEN",'
                '"severity":"CRITICAL","cfr":"FALCPA, 21 CFR 101.4",'
                '"observed":"milk not declared","expected":"milk must be declared",'
                '"correction":"declare milk in ingredients or Contains"}]}'
            ),
        }
    ]
