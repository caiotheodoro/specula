from specula_model.train_config import (
    SMOKE_MAX_STEPS,
    dummy_sft_records,
    sft_records_from_bytes,
    trainer_kwargs,
)


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
        assert "FDA" in turns[0]["content"]
        assert '"verdict"' in turns[1]["content"]


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
