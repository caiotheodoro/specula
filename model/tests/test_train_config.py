import json

from specula_model.train_config import (
    SMOKE_MAX_STEPS,
    dummy_sft_records,
    grpo_kwargs,
    grpo_trainer_kwargs,
    rlvr_records_from_bytes,
    sft_records_from_bytes,
    trainer_kwargs,
)

import pytest



def test_smoke_trainer_kwargs_use_4bit_and_capped_steps():
    kwargs = trainer_kwargs(smoke=True)
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_steps"] == SMOKE_MAX_STEPS == 4
    assert kwargs["num_train_epochs"] == 1
    assert kwargs["max_seq_length"] == 256
    assert kwargs["loss_type"] == "nll"


def test_full_trainer_kwargs_keep_4bit_and_honor_epochs():
    kwargs = trainer_kwargs(smoke=False, epochs=3)
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_steps"] == -1
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["max_seq_length"] == 1024
    assert kwargs["loss_type"] == "nll"


def test_dummy_sft_records_are_trl_conversational():
    records = dummy_sft_records()
    assert records
    for record in records:
        turns = record["messages"]
        assert [turn["role"] for turn in turns] == ["user", "assistant"]
        user = turns[0]["content"]
        text = next(part["text"] for part in user if part.get("type") == "text")
        assert {"type": "image"} in user
        assert record["images"]
        assert "FDA" in text
        asst = turns[1]["content"]
        assert asst[0]["type"] == "text"
        assert '"verdict"' in asst[0]["text"]


def test_empty_bytes_fall_back_to_dummy_records():
    assert sft_records_from_bytes(b"") == dummy_sft_records()


def test_dataset_builder_jsonl_becomes_trl_messages():
    line = (
        '{"user_text":"Review this food label for FDA compliance. ",'
        '"assistant_json":"{\\"verdict\\":\\"PASS\\",\\"violations\\":[]}"}\n'
    )
    records = sft_records_from_bytes(line.encode())
    assert records[0]["messages"][0]["role"] == "user"
    assert "FDA" in records[0]["messages"][0]["content"]
    assert '"verdict"' in records[0]["messages"][1]["content"]


def test_forge_task_jsonl_uses_expected_verdict():
    line = (
        '{"task_id":"t1","expected":{"verdict":"FLAG","violations":[]}}\n'
    )
    records = sft_records_from_bytes(line.encode())
    assert records[0]["messages"][1]["role"] == "assistant"
    assert '"FLAG"' in records[0]["messages"][1]["content"]


def test_dataset_builder_jsonl_attaches_vl_image():
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    line = json.dumps({
        "user_text": "Review this food label for FDA compliance. ",
        "assistant_json": '{"verdict":"PASS","violations":[]}',
        "image_b64": png_b64,
    }) + "\n"
    records = sft_records_from_bytes(line.encode())
    user = records[0]["messages"][0]["content"]
    assert {"type": "image"} in user
    assert any(part.get("type") == "text" and "FDA" in part["text"] for part in user)
    assert records[0]["images"] == [png_b64]
    asst = records[0]["messages"][1]["content"]
    assert asst[0]["type"] == "text"
    assert '"PASS"' in asst[0]["text"]



def test_smoke_grpo_kwargs_are_dr_grpo_with_dapo_clip():
    kwargs = grpo_kwargs(smoke=True)
    assert kwargs["loss_type"] == "dr_grpo"
    assert kwargs["epsilon"] == 0.2
    assert kwargs["epsilon_high"] == 1.0
    assert kwargs["num_generations"] == 2
    assert kwargs["per_device_train_batch_size"] % kwargs["num_generations"] == 0
    assert kwargs["max_steps"] == 2
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_completion_length"] <= 64


def test_full_grpo_kwargs_honor_iters_and_group_size():
    kwargs = grpo_kwargs(smoke=False, iters=50, group_size=4)
    assert kwargs["loss_type"] == "dr_grpo"
    assert kwargs["num_generations"] == 4
    assert kwargs["max_steps"] == 50
    assert kwargs["per_device_train_batch_size"] % 4 == 0


def test_dummy_rlvr_records_have_no_assistant_gold():
    records = rlvr_records_from_bytes(b"")
    assert records
    for rec in records:
        roles = [turn["role"] for turn in rec["prompt"]]
        assert "assistant" not in roles
        assert "expected_verdict" in rec
        assert rec["images"]


def test_rlvr_records_reject_sft_assistant_json():
    line = json.dumps({
        "user_text": "Review this food label",
        "assistant_json": '{"verdict":"PASS","violations":[]}',
    }) + "\n"
    with pytest.raises(ValueError, match="SFT"):
        rlvr_records_from_bytes(line.encode())


def test_grpo_trainer_kwargs_drop_unknown_and_bnb_flag():
    kw = grpo_kwargs(smoke=True)
    accepted = {"loss_type", "epsilon", "epsilon_high", "num_generations",
                "max_completion_length"}
    out = grpo_trainer_kwargs(kw, accepted)
    assert "load_in_4bit" not in out
    assert "max_prompt_length" not in out
    assert out["loss_type"] == "dr_grpo"
    assert out["num_generations"] == 2
